
use crate::mat;
use std::sync::Arc;
use crate::mat::{Mat, V3, DMat, V};
use rand::{Rng, SeedableRng, thread_rng};
use rand_xoshiro::SplitMix64;
use rand_distr::Standard;
use std::f32::consts::PI;
use std::collections::VecDeque;
use std::mem::swap;
use std::ops::Mul;
use scoped_threadpool::Pool;
use std::any::Any;
use num;

/// A complex number
pub type Complex = num::Complex<f32>;

/// Color / channel intensities in calculations.
/// Each color channel is handled separately.
pub type Color = V3;

/// Material applicable to a point on a surface.
#[derive(Clone)]
pub struct PointMaterial {
    /// The color (~albedo) of this point.
    pub color : Color,

    /// The amount of light scattering from this point.
    ///
    /// 0.0 means no scattering, higher numbers increase the randomness
    /// of each scattered ray.
    pub roughness : f32,

    /// The emissive component.
    pub emissive : Color,

    /// The diffuse / normal reflection multiplier.
    ///
    /// `0.0` to disable diffuse bounces entirely (e.g. a purely emissive material).
    /// Note that this also affects transmission.
    pub diffuse : f32,

    /// The transmission / scattering multiplier
    ///
    /// `0.0` means no transmission at all (all rays are considered to scatter into the surface layer),
    /// `1.0` corresponds to a fully transparent material (no diffusion). Note that this is also
    /// scaled according to the amount of light that is refracted "into" the material.
    pub transmission: f32,
    
    /// The (possibly complex) index of refraction of the material.
    ///
    /// This accounts for the Fresnel effect for high-angle incident rays
    /// as well as for refraction.
    pub ior : Complex
}

/// A sampling ray in the scene
#[derive(Debug)]
pub struct Ray {
    /// The starting point of the ray
    origin : V3,
    /// The direction of the ray (normalized)
    direction : V3,
    /// The number of bounces that have been made by rays up to this one
    bounces : i32,
    /// The number of splits that have been made by rays up to this one
    splits: i32,
    /// The (approximate) contribution of this ray to the final pixel
    contribution : f32
}

/// The properties of a ray hitting a surface
#[derive(Clone)]
pub struct RayHit {
    /// The distance from the origin of the ray to the hit position
    distance : f32,
    
    /// The position where the hit occurred
    position : V3,
    
    /// The normal of the surface at the hit point
    normal : V3,
    
    /// The material of the hit point
    material : Arc<PointMaterial>
}

/// A transform matrix, to be left-multiplied to positions (column vectors)
pub type Transform = Mat<4, 4, f32>;

impl Transform {
    /// The identity transform
    const IDENTITY : Transform = Mat::new([
	[1.0, 0.0, 0.0, 0.0],
	[0.0, 1.0, 0.0, 0.0],
	[0.0, 0.0, 1.0, 0.0],
	[0.0, 0.0, 0.0, 1.0]
    ]);

    /// Generate a transform describing a translation by a vector
    pub fn translation(vec : V3) -> Transform {
	let mut t = Self::IDENTITY;
	for i in 0..3 {
	    t[i][3] = vec[i];
	}
	t
    }

    /// Generate a transform sequentially rotating about the x, y and z axes
    ///
    /// This uses a right-hand coordinate system, rotations are counterclockwise
    /// looking along the corresponding unit vector
    pub fn rotation_xyz(x : f32, y : f32, z : f32) -> Transform {
	let mut t = Self::IDENTITY;

	if x != 0.0 {
	    let s = x.sin();
	    let c = x.cos();
	    t = mat![
		[1.0, 0.0, 0.0, 0.0],
		[0.0,   c,   s, 0.0],
		[0.0,  -s,   c, 0.0],
		[0.0, 0.0, 0.0, 1.0]
	    ];
	}

	if y != 0.0 {
	    let s = y.sin();
	    let c = y.cos();
	    t = mat![
		[  c, 0.0,  -s, 0.0],
		[0.0, 1.0, 0.0, 0.0],
		[  s, 0.0,   c, 0.0],
		[0.0, 0.0, 0.0, 1.0]
	    ] * t;
	}

	if z != 0.0 {
	    let s = z.sin();
	    let c = z.cos();
	    t = mat![
		[  c,   s, 0.0, 0.0],
		[ -s,   c, 0.0, 0.0],
		[0.0, 0.0, 1.0, 0.0],
		[0.0, 0.0, 0.0, 1.0]
	    ] * t;
	}
	
	t
    }

    /// Generate a vector describing scaling the system along the axes
    /// corresponding to the input vector
    pub fn scale(vec : V3) -> Transform {
	let mut t = Self::IDENTITY;

	for i in 0..3 {
	    t[i][i] = vec[i];
	}
	t
    }

    /// Transform a V3 by this transform
    pub fn v3(self, vec : &V3) -> V3 {
	let mut v = vec.slice::<4,1>();
	v[3] = 1.0;
	(self * v).slice::<3,1>()
    }
}


impl Mul<&V3> for Transform {
    type Output = V3;

    fn mul(self, vec : &V3) -> V3 {
	self.v3(vec)
    }
}

/// A renderable object
pub trait Renderable : Sync + Send {
    /// Allow calculating a cache (for each [`SceneObject`]) before rendering
    fn calc_cache(&self, object : &SceneObject) -> Box<dyn Any + Send + Sync>;

    /// Raycast against this renderable
    /// The closest (if any) hit for this renderable will be returned
    fn raycast(&self, object : &SceneObject, ray : &Ray, flipped : bool) -> Option<RayHit>;
}

/// Get the vector from the ray's origin to the point
/// along with the distance between the corresponding line and the point
fn point_ray(point : V3, ray : &Ray) -> (V3, f32) {
    let x = point - ray.origin;
    let n = ray.direction.cross(x);
    let l2 = n.sq_magnitude();

    (x, l2)
}

/// A perfect sphere centered on `[0.0, 0.0, 0.0]` with a radius of 0.5
pub struct Sphere {
    /// The material to use for each point
    pub material: Arc<PointMaterial>
}

struct SphereCache {
    pos : V3,
    r2 : f32
}

impl Renderable for Sphere {
    fn raycast(&self, object : &SceneObject, ray : &Ray, flipped : bool) -> Option<RayHit> {
	// Extract the position and the square of the radius from the cache
	
	let cache : &SphereCache = object.cache.as_ref().unwrap().downcast_ref().unwrap();
	let flip = if flipped { -1.0 } else { 1.0 };
	
	// Check whether the line corresponding to the ray intersects the sphere
	let (x, l2) = point_ray(cache.pos, ray);
	if l2 <= cache.r2 {
	    // Check whether the correct intersection with the line occurs on the ray
	    let dist = x.dot(ray.direction) - (cache.r2 - l2).sqrt() * flip;
	    if dist > 0.0 {
		// A valid hit
		let position = ray.origin + ray.direction * dist;
		let normal = (position - cache.pos).normalize() * flip;
		return Some(RayHit {
		    distance: dist,
		    position,
		    normal,
		    material: self.material.clone()
		});
	    }
	}
	None
    }

    fn calc_cache(&self, object : &SceneObject) -> Box<dyn Any + Send + Sync> {
	let pos = object.transform.v3(&V3::zero());
	Box::new(SphereCache {
	    pos,
	    r2: (object.transform.v3(&mat![0.5, 0.0, 0.0]) - pos).sq_magnitude()
	})
    }
}

/// A mesh consisting of triangles
pub struct SimpleMesh {
    /// The material used for the mesh
    pub material : Arc<PointMaterial>,
    
    /// The triangles of this mesh
    /// Each triangle is specified in counterclockwise order
    pub tris : Vec<[V3; 3]>
}

struct SimpleMeshCache {
    bounding_sphere : BoundingSphere,
    tris : Vec<TriCache>
}

impl Renderable for SimpleMesh {
    fn raycast(&self, object : &SceneObject, ray : &Ray, flipped : bool) -> Option<RayHit> {
	// Extract the transformed triangles and the bounding sphere from the cache
	let cache : &SimpleMeshCache = object.cache.as_ref().unwrap().downcast_ref().unwrap();

	let flip = if flipped { -1.0 } else { 1.0 };
	
	// Check whether the ray collides with the bounding sphere
	let (_, l2) = point_ray(cache.bounding_sphere.pos, ray);
	if l2 > cache.bounding_sphere.r2 {
	    return None;
        }

	// Iterate over the triangles to find a hit
	let mut best : Option<(V3, f32, V3)> = None;
	for t in cache.tris.iter() {
	    // Backface culling - can't see a triangle oriented the wrong way
	    let dot = t.normal.dot(ray.direction) * flip;
	    if dot < 0.0 {
		// Perpendicular distance
		let pdist = t.normal.dot(ray.origin - t.a) * flip;
		if pdist > 0.0 {
		    let dist = pdist / -dot;
		    
		    if let Some((_, distance, _)) = &best {
			if *distance <= dist {
			    // There is already a better hit
			    continue;
			}
		    }
		    
		    let point = ray.origin + ray.direction * dist;
		    
		    // Now check whether this point is actually *inside* the triangle
		    if t.ab_i.dot(point - t.a) >= 0.0 &&
			t.bc_i.dot(point - t.b) >= 0.0 &&
			t.ca_i.dot(point - t.c) >= 0.0 {
			    // New best hit
			    best = Some((point, dist, t.normal))
			}
		}
	    }
	}
	match best {
	    None => None,
	    Some((position, distance, normal)) =>
		Some(RayHit {
		    distance,
		    material: self.material.clone(),
		    position,
		    normal: normal * flip
		})
	}
    }

    fn calc_cache(&self, object : &SceneObject) -> Box<dyn Any + Send + Sync> {
	let tris : Vec<TriCache> = self.tris.iter().map(
	    |[a, b, c]| calc_tri([
		object.transform.v3(a),
		object.transform.v3(b),
		object.transform.v3(c)
	    ])).collect();

	let mid : V3 = tris.iter().map(|t| t.a + t.b + t.c)
	    .fold(V3::zero(), |a, b| a + b) * (1.0 / (3 * tris.len()) as f32);
	let r2 = tris.iter().map(
	    |t| (t.a - mid).sq_magnitude().max((t.b - mid).sq_magnitude()).max((t.c - mid).sq_magnitude())
	).fold(0.0f32, |a, b| a.max(b));
	Box::new(SimpleMeshCache {
	    tris,
	    bounding_sphere: BoundingSphere { pos: mid, r2 }
	})
    }
}

/// Triangle cache
#[derive(Debug, Clone)]
struct TriCache {
    a : V3,
    b : V3,
    c : V3,
    normal : V3,
    ab_i : V3,
    bc_i : V3,
    ca_i : V3
}

/// Calculate the cache for a single triangle
fn calc_tri(raw : [V3; 3]) -> TriCache {
    let [a, b, c] = raw;
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    let normal = (-ab.cross(ca)).normalize();
    TriCache {
	a, b, c, 
	normal,
	ab_i: normal.cross(ab),
	bc_i: normal.cross(bc),
	ca_i: normal.cross(ca)
    }
}

/// A mesh consisting of triangles with a coordinate mapping
pub struct CoordinateMesh<const N : usize> {
    /// The function used to assign a material given the coordinates
    pub material_fn : Arc<dyn Fn(V<N, f32>,) -> PointMaterial + Sync + Send>,
    
    /// The triangles of this mesh
    /// Each triangle is specified in counterclockwise order
    pub tris : Vec<[V3; 3]>,

    /// The coordinates for each vertex of each triangle specified in `tris`
    pub coords : Vec<[V<N, f32>; 3]>
}

struct CoordinateMeshCache<const N : usize> {
    bounding_sphere : BoundingSphere,
    tris : Vec<CoordTriCache<N>>
}

impl<const N : usize> Renderable for CoordinateMesh<N> {
    fn raycast(&self, object : &SceneObject, ray : &Ray, flipped : bool) -> Option<RayHit> {
	// Extract the transformed triangles and the bounding sphere from the cache
	let cache : &CoordinateMeshCache<N> = object.cache.as_ref().unwrap().downcast_ref().unwrap();

	let flip = if flipped { -1.0 } else { 1.0 };
	
	// Check whether the ray collides with the bounding sphere
	let (_, l2) = point_ray(cache.bounding_sphere.pos, ray);
	if l2 > cache.bounding_sphere.r2 {
	    return None;
        }

	// Iterate over the triangles to find a hit
	let mut best : Option<(V3, f32, V3, V<N, f32>)> = None;
	for t in cache.tris.iter() {
	    // Backface culling - can't see a triangle oriented the wrong way
	    let dot = t.normal.dot(ray.direction) * flip;
	    if dot < 0.0 {
		// Perpendicular distance
		let pdist = t.normal.dot(ray.origin - t.a) * flip;
		if pdist > 0.0 {
		    let dist = pdist / -dot;
		    
		    if let Some((_, distance, _, _)) = &best {
			if *distance <= dist {
			    // There is already a better hit
			    continue;
			}
		    }
		    
		    let point = ray.origin + ray.direction * dist;
		    
		    // Now check whether this point is actually *inside* the triangle
		    let ap = point - t.a;
		    if t.ab_i.dot(ap) >= 0.0 {
			let bp = point - t.b;
			if t.bc_i.dot(bp) >= 0.0 {
			    let cp = point - t.c;
			    if t.ca_i.dot(cp) >= 0.0 {
				// New best hit
				// Calculate the coordinates
				// Scale by (height from vertex) / (max height from vertex)
				let p_a = (t.a_h + t.bc_i.dot(ap)) / t.a_h;
				let p_b = (t.b_h + t.ca_i.dot(bp)) / t.b_h;
				let p_c = (t.c_h + t.ab_i.dot(cp)) / t.c_h;
				
				let coordinates = (t.a_c * p_a + t.b_c * p_b  + t.c_c * p_c)
				    * (1.0 / (p_a + p_b + p_c));
				best = Some((point, dist, t.normal, coordinates))
			    }
			}
		    }
		}
	    }
	}
	match best {
	    None => None,
	    Some((position, distance, normal, coordinates)) =>
		Some(RayHit {
		    distance,
		    material: (self.material_fn)(coordinates).into(),
		    position,
		    normal: normal * flip
		})
	}
    }

    fn calc_cache(&self, object : &SceneObject) -> Box<dyn Any + Send + Sync> {
	let tris : Vec<CoordTriCache<N>> = self.tris.iter().zip(self.coords.iter()).map(
	    |([a, b, c], coords)| calc_coord_tri([
		object.transform.v3(a),
		object.transform.v3(b),
		object.transform.v3(c)
	    ], coords)).collect();

	let mid : V3 = tris.iter().map(|t| t.a + t.b + t.c)
	    .fold(V3::zero(), |a, b| a + b) * (1.0 / (3 * tris.len()) as f32);
	let r2 = tris.iter().map(
	    |t| (t.a - mid).sq_magnitude().max((t.b - mid).sq_magnitude()).max((t.c - mid).sq_magnitude())
	).fold(0.0f32, |a, b| a.max(b));
	Box::new(CoordinateMeshCache {
	    tris,
	    bounding_sphere: BoundingSphere { pos: mid, r2 }
	})
    }
}

/// Triangle cache
#[derive(Debug, Clone)]
struct CoordTriCache<const N : usize> {
    a : V3,
    b : V3,
    c : V3,
    normal : V3,
    ab_i : V3,
    bc_i : V3,
    ca_i : V3,
    a_h : f32,
    b_h : f32,
    c_h : f32,
    a_c : V<N, f32>,
    b_c : V<N, f32>,
    c_c : V<N, f32>
}

/// Calculate the cache for a single triangle
fn calc_coord_tri<const N : usize>(raw : [V3; 3], coords : &[V<N, f32>; 3]) -> CoordTriCache<N> {
    let [a, b, c] = raw;
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    let normal = (-ab.cross(ca)).normalize();
    let ab_i = normal.cross(ab).normalize();
    let bc_i = normal.cross(bc).normalize();
    let ca_i = normal.cross(ca).normalize();
    CoordTriCache {
	a, b, c,
	normal,
	ab_i, bc_i, ca_i,
	a_h: bc_i.dot(ca),
	b_h: ca_i.dot(ab),
	c_h: ab_i.dot(bc),
	a_c: coords[0],
	b_c: coords[1],
	c_c: coords[2]
    }
}


/// Render-specific properties
/// 
/// Note that some of the properties affecting rendering are also
/// present in [`Camera`] and [`Scene`] 
#[derive(Clone)]
pub struct RenderProperties {
    /// The resolution (`[width, height]`) of the render in pixels
    pub resolution : [i32; 2],
    
    /// The number of rendering passes to perform
    pub passes : i32,

    /// The maximum number of bounces to perform for each ray
    pub bounces : i32,
    
    /// How many rays to split a single ray into when randomly bouncing
    pub bounce_split : i32,

    /// How many times to allow a ray to be split into multiple
    pub split_cap : i32,
    
    /// How many threads to use for rendering
    pub threads : i32,

    /// Approximately how many pixels a single thread should try to render
    pub pixels_per_thread : i32
}

/// The fraction of light falling onto the surface to reflect (directly).
fn p_reflection(n1 : Complex, n2 : Complex, cos_inc : f32) -> f32 {
    // Approximate
    // https://en.wikipedia.org/wiki/Schlick%27s_approximation
    let r0 = ((n1 - n2) / (n1 + n2)).norm_sqr();
    
    r0 + (1.0 - r0) * (1.0 - r0) * (1.0 - cos_inc).powi(5)
	
    /*
    // Exact
    // https://en.wikipedia.org/wiki/Fresnel_equations
    let s = (Complex::from(1.0) - (n1 / n2).powi(2) * Complex::from(1.0 - cos_inc.powi(2))).sqrt();
    let u1 : Complex = n1 * cos_inc;
    let u2 : Complex = n2 * s;
    let v1 : Complex = n1 * s;
    let v2 : Complex = n2 * cos_inc;
    
    (((u1 - u2) / (u1 + u2)).norm_sqr() +
	((v1 - v2) / (v1 + v2)).norm_sqr()) / 2.0
    */
}

/// Calculate the vector pointing in the direction the refracted ray should go
fn refraction_direction(normal : V3, incident : V3, cos_inc : f32, n1 : Complex, n2 : Complex) -> V3 {
    let sin_inc = (1.0 - cos_inc.powi(2)).sqrt();
    let sin_refract = n1.re / n2.re * sin_inc;
    // Direction along the surface in the direction of the ray
    let dir_tangent = (incident - normal * normal.dot(incident))
	.normalize();
    let cos_refract = (1.0 - sin_refract.powi(2)).sqrt();
    dir_tangent * sin_refract - normal * cos_refract
}


/// A medium through which rays can pass
#[derive(Clone)]
struct Medium<'a, 'b> {
    ior : Complex,
    object : Option<&'a SceneObject>,
    previous : Option<&'b Medium<'a, 'b>>
}

const INTENSITY_EPSILON : f32 = 1e-10;
const DISTANCE_EPSILON : f32 = 1e-6;

/// Handle bounces / refractions at a point with the given normal
fn handle_ray_with_normal(render_props : &RenderProperties, scene : &Scene, randomness : &mut Randomness,
			  ray : &Ray, hit : &RayHit, medium : &Medium, next_medium : &Medium,
			  normal : V3, split_ray : bool) -> Color {
    let mut color = Color::zero();
    
    // Get the incident angle to the normal
    let cos_inc = -normal.dot(ray.direction);

    // Find the proportion of light that is reflected
    let n1 = medium.ior;
    let n2 = hit.material.ior;
    let p_reflect = p_reflection(n1, n2, cos_inc);
    
    // Don't calculate reflections for rays that are purely transmitted
    if p_reflect > INTENSITY_EPSILON || hit.material.transmission != 1.0 {
	// Reflected direction
	let direction = normal * (2.0 * cos_inc) + ray.direction;
	
	// Render the reflected ray
	let fall = render_ray(render_props, scene, randomness, Ray {
	    origin: hit.position + direction * DISTANCE_EPSILON,
	    direction,
	    bounces: ray.bounces + 1,
	    splits: ray.splits + if split_ray { 1 } else { 0 },
	    contribution: ray.contribution * p_reflect
	}, medium);
	
	// Reflected through
	color += fall * p_reflect;
	
	// Reflected channel-wise
	color += fall.dot_matrix(hit.material.color) * (1.0 - p_reflect)
	    * (1.0 - hit.material.transmission);
    }

    // Don't calculate transmission for rays that are almost entirely reflected
    let p_transmit = (1.0 - p_reflect) * hit.material.transmission;

    if p_transmit > INTENSITY_EPSILON {
	
	let dir_refract = refraction_direction(normal, ray.direction,
					       cos_inc, medium.ior, hit.material.ior);
	
	let transmission =  render_ray(render_props, scene, randomness, Ray {
	    origin: hit.position + dir_refract * DISTANCE_EPSILON,
	    direction: dir_refract,
	    bounces: ray.bounces + 1,
	    splits: ray.splits + if split_ray { 1 } else { 0 },
	    contribution: ray.contribution * p_transmit
	}, next_medium);
	color += transmission * p_transmit;
    }

    color
}

/// Render the diffuse (scatter + albedo + transmission) component of a ray falling onto a surface
fn render_ray_diffuse(render_props : &RenderProperties, scene : &Scene, randomness : &mut Randomness,
		      ray : Ray, hit : RayHit, medium : &Medium, next_medium : &Medium,
		      exit_hit : Option<RayHit>) -> Color {
    let mut color = Color::zero();
    
    // Emission
    color += hit.material.emissive;

    if hit.material.diffuse != 0.0 && ray.bounces < render_props.bounces {
	let mut diffuse = Color::zero();
	
	let cos_inc = -hit.normal.dot(ray.direction);
	let perfect = hit.normal * (2.0 * cos_inc) + ray.direction;

	let roughness = match exit_hit {
	    Some(ref exit_hit) => hit.material.roughness.max(exit_hit.material.roughness),
	    None => hit.material.roughness
	};
	
	if roughness == 0.0 {
	    // No need to create multiple rays here (they would all bounce the same)
	    diffuse += handle_ray_with_normal(render_props, scene, randomness, &ray, &hit,
					      medium, next_medium, hit.normal, false);
	} else {
	    // Perturbed ray
	    let num_rays = if ray.splits < render_props.split_cap {
		render_props.bounce_split } else { 1 };
	    for _ in 0..num_rays {
		// Generate a ray that does not go into the plane
		let direction = {
		    // Cylindrical coordinates
		    let t = 2.0 * PI * randomness.rng.gen::<f32>();
		    let y = 2.0 * randomness.rng.gen::<f32>() - 1.0;

		    // Point on the unit sphere
		    let r = (1.0 - y.powi(2)).sqrt();
		    let x = r * t.cos();
		    let z = r * t.sin();

		    // Perturbed direction
		    // Since the perfect ray has a positive dot product with the normal,
		    // either the perturbed direction or the negatively perturbed direction
		    // will have a positive dot product as well
		    let a = mat![x, y, z] * roughness;
		    let dir = perfect + a;
		    if dir.dot(hit.normal) > 0.0 {
			dir
		    } else {
			perfect - a
		    }
		}.normalize();

		// Extract the resulting normal vector
		let local_normal = (direction - ray.direction).normalize();
		diffuse += handle_ray_with_normal(render_props, scene, randomness, &ray, &hit,
						  medium, next_medium, local_normal, num_rays > 1);
	    }

	    diffuse *= 1.0 / (num_rays as f32);
	}

	color += diffuse * hit.material.diffuse;
    }
    
    color
}
    
/// Render a ray in the scene
fn render_ray(render_props : &RenderProperties, scene : &Scene, randomness : &mut Randomness, ray : Ray,
	      medium : &Medium) -> Color {
    if ray.contribution < INTENSITY_EPSILON {
	return Color::zero();
    }
    // Find the closest ray hit
    let mut best : Option<(RayHit, &SceneObject)> = None;
    for obj in scene.objects.iter() {
	if let Some(curr) = obj.renderable.raycast(&obj, &ray, false) {
	    match best {
		Some((ref b, _)) => {
		    if b.distance > curr.distance {
			best = Some((curr, obj));
		    }
		},
		None => best = Some((curr, obj))
	    }
	}
    }

    // Check whether we have reached the inner boundary of the current medium
    // Exiting an *outer* medium before this is UB (physics-wise)
    if let Some(medium_object) = medium.object {
	if let Some(inner_hit) = medium_object.renderable.raycast(medium_object, &ray, true) {
	    if let Some((ref hit, _)) = best {
		if (inner_hit.distance - hit.distance).abs() < DISTANCE_EPSILON {
		    // These surfaces seem to be squished together
		    // Note that this can't currently handle exiting several mediums at once
		    let prev_medium = medium.previous.unwrap();
		    return render_ray_diffuse(render_props, scene, randomness, ray, hit.clone(),
					      medium, &Medium {
						  ior: hit.material.ior,
						  object: prev_medium.object,
						  previous: prev_medium.previous
					      }, Some(inner_hit));
		} else if inner_hit.distance < hit.distance {
		    // Boundary of the current medium
		    return render_ray_diffuse(render_props, scene, randomness, ray, inner_hit,
					      medium, medium.previous.unwrap(), None);
		}
	    } else {
		// Nothing else to hit
		// Boundary of the current medium
		return render_ray_diffuse(render_props, scene, randomness, ray, inner_hit,
					  medium, medium.previous.unwrap(), None);
	    }
	}
    }

    if let Some((hit, object)) = best {
	// Handle the hit
	let next_medium = Medium {
	    ior: hit.material.ior,
	    object: Some(object),
	    previous: Some(medium),
	};
	render_ray_diffuse(render_props, scene, randomness, ray, hit, medium, &next_medium, None)
    } else {
	// Ray didn't hit anything - sky
	scene.sky.sky_ray(&ray)
    }
}

/// The randomness provider for rendering
#[derive(Clone)]
struct Randomness {
    rng : SplitMix64,
}

impl Randomness {
    fn new() -> Randomness{
	let rng = SplitMix64::from_rng(thread_rng()).unwrap();
	Randomness {
	    rng
	}
    }
}

/// A camera positioned in the scene
pub struct Camera {
    /// The transform of the camera
    pub transform : Transform,

    /// The horizontal FOV of the camera in radians
    pub h_fov : f32
}

impl Camera {
    /// Render a single pixel
    fn render_px(&self, render_props : &RenderProperties, scene : &Scene, randomness : &mut Randomness,
		 point : [i32; 2]) -> Color {
	// Screen space + noise
	let x = (point[0] as f32 + randomness.rng.sample::<f32, Standard>(Standard) - 0.5)
	    / render_props.resolution[0] as f32;
	let y = (point[1] as f32 + randomness.rng.sample::<f32, Standard>(Standard) - 0.5)
	    / render_props.resolution[1] as f32;
	
	// Local space, 1 unit from the camera origin in -Z
	let vw = (self.h_fov / 2.0).tan() * 2.0;
	let vh = vw * render_props.resolution[1] as f32 / render_props.resolution[0] as f32;
	let pos = mat![vw * (x - 0.5), vh * (0.5 - y), -1.0];

	// World space
	let pos = self.transform.v3(&pos);
	let origin = self.transform.v3(&V3::zero());
	let direction = (pos - origin).normalize();

	render_ray(render_props, scene, randomness, Ray {
	    origin,
	    direction,
	    bounces: 0,
	    splits: 0,
	    contribution: 1.0
	}, &Medium {
	    ior: scene.ior,
	    object: None,
	    previous: None
	})
    }

    /// Convert a possible [`Mat`] of some size to one of the required size
    fn ensure_mat(mat : Option<DMat<Color>>, size : [i32; 2]) -> DMat<Color> {
	let sz = [size[1] as usize, size[0] as usize];
	match mat {
	    Some(mut m) => {m.resize(sz, Color::zero()); m},
	    None => DMat::new(sz, Color::zero())
	}
    }

    /// Render a rectangle of pixels
    fn render_rectangle(&self, render_props : &RenderProperties, scene : &Scene,
			randomness : &mut Randomness, start : [i32; 2],
			stop : [i32; 2], slice : &mut [Color])  {
	let w = stop[0] - start[0];
	let h = stop[1] - start[1];

	assert!((w * h) as usize <= slice.len());
	
	for y in 0..h {
	    for x in 0..w {
		slice[(y * w + x) as usize] =
		    self.render_px(render_props, scene, randomness, [x + start[0], y + start[1]]);
	    }
	}
    }


    /// Render a single pass of the entire image
    pub fn render_pass(&self, render_props : &RenderProperties,
		   scene : &Scene, mat : Option<DMat<Color>>) -> DMat<Color> {
	// Ensure we have a mat to store the result in
	let mut res = Self::ensure_mat(mat, render_props.resolution);

	// Create a randomness provider
	if render_props.threads <= 1 {
	    // Single-threaded render
	    let mut randomness = Randomness::new();
	    self.render_rectangle(render_props, scene, &mut randomness,
				  [0, 0], render_props.resolution, &mut res.data);
	} else {
	    // Multithreaded render
	    let [w, h] = render_props.resolution;

	    // Split the mat into full-width rectangles to be rendered
	    let mut cy = 0;
	    let sy = (render_props.pixels_per_thread / w).max(1);
	    let res_slice : &mut [Color] = &mut res.data;
	    
	    // Scoping + threading time
	    let mut pool = Pool::new(render_props.threads as u32);
	    pool.scoped(|scoped| {
		for chunk in res_slice.chunks_mut((sy * w) as usize) {
		    let ey = (cy + sy).min(h);
		    
		    scoped.execute(move || {
			// Generate randomness (thread-local)
			let mut randomness = Randomness::new();
			// Render
			self.render_rectangle(render_props, scene, &mut randomness,
					      [0, cy], [w, ey], chunk);
		    });
		    
		    cy = ey;
		}
	    });
	}
	res
    }

    /// Render an image
    pub fn render(&self, render_props : &RenderProperties, scene : &Scene) -> DMat<Color> {
	let [w, h] = render_props.resolution;
	let mut mats = VecDeque::new();
	let mut curr = DMat::new([w as usize, h as usize], Color::zero());

	if render_props.passes < 1 {
	    // ??
	    return curr;
	}
	
	// Render, combining frames by powers of 2
	for i in 1..=render_props.passes {
	    curr = self.render_pass(render_props, scene, Some(curr));

	    // Sum
	    let mut b = 0;
	    while i & (1 << b) == 0 {
		// Add to this value, propagate curr
		let mat = &mut mats[b as usize];
		*mat += &curr;
		swap(&mut curr, mat);
		b += 1;
	    }
	    // Assign to this index
	    // Check whether this index exists
	    if b == mats.len() {
		// Does not exist - push this and create a new one
		mats.push_back(curr);
		curr = DMat::new([w as usize, h as usize], Color::zero());
	    } else {
		// Exists - swap
		swap(&mut curr, &mut mats[b]);
	    }
	}

	// Collect remaining frames
	let mut b = render_props.passes;
	let mut curr = None;
	while mats.len() > 0 {
	    let m = mats.pop_front().unwrap();
	    if (b & 1) != 0 {
		// Add this frame
		curr = if let Some(mut c) = curr {
		    c += &m;
		    Some(c)
		} else {
		    // No existing frame
		    Some(m)
		}
	    }
	    b >>= 1;
	}
	
	// Normalize the result
	let mut curr = curr.unwrap();
	for v in curr.data.iter_mut() {
	    *v *= 1.0 / (render_props.passes as f32);
	}
	
	curr
    }
}

/// An instance of an object in the scene
pub struct SceneObject {
    pub transform : Transform,
    pub renderable : Arc<dyn Renderable>,
    cache : Option<Box<dyn Any + Send + Sync>>
}

impl SceneObject {
    pub fn new(transform : Transform, renderable: Arc<dyn Renderable>) -> SceneObject {
	SceneObject {
	    transform,
	    renderable,
	    cache: None
	}
    }
}


/// A bounding sphere of an object
pub struct BoundingSphere {
    /// The center of the bounding sphere
    pos : V3,
    
    /// The square of the radius of the bounding sphere
    r2 : f32
}

/// A patch of some material
pub struct Patch {
    /// The firection on which this patch is centered
    pub direction : V3,

    /// The minimum cosine of the angle from the `direction`
    /// where the patch exists
    pub min_cos : f32,

    /// The color (emissive intensities) of the patch
    pub color : Color
}

impl Patch {
    /// Check if a ray hits this patch
    fn ray(&self, ray : &Ray) -> Option<Color> {
	let cos = self.direction.dot(ray.direction);
	if cos >= self.min_cos {
	    Some(self.color)
	} else {
	    None
	}
    }
}

/// The sky (background) of a scene
#[derive(Default)]
pub struct Sky {
    /// The base color (emissive intensity) of the sky
    pub color : Color,

    /// The patches present in the sky, in order of importance
    pub patches : Vec<Patch>
}

impl Sky {
    /// Get the intensities of a ray hitting the sky
    fn sky_ray(&self, ray : &Ray) -> Color {
	for patch in self.patches.iter() {
	    if let Some(color) = patch.ray(ray) {
		return color;
	    }
	}
	self.color
    }
}

/// A scene holding the objects to be rendered
#[derive(Default)]
pub struct Scene {
    pub objects : Vec<SceneObject>,
    pub sky : Sky,
    pub ior : Complex
}

impl Scene {
    /// Calculate the cache for the scene
    pub fn calc_cache(&mut self) {
	for obj in self.objects.iter_mut() {
	    obj.cache = Some(obj.renderable.calc_cache(&obj));
	}
    }
}

/// Construct a triangle mesh from points
///
/// Every three consecutive points form a triangle, with the order swapping between
/// counterclockwise and clockwise after every triangle (starting counterclockwise)
pub fn triangle_march<const N : usize>(points : &Vec<V<N, f32>>) -> Vec<[V<N, f32>; 3]> {
    let mut r = Vec::new();
    
    if points.len() > 2 {
	let mut iter = points.iter();
	let mut a = iter.next().unwrap();
	let mut b = iter.next().unwrap();

	let mut flip = false;
	
	while let Some(p) = iter.next() {
	    r.push(if flip {
		[*b, *a, *p]
	    } else {
		[*a, *b, *p]
	    });
	    a = b;
	    b = p;
	    flip = !flip;
	}
    }
    r
}
