
mod mat;
mod renderer_core;

use renderer_core::*;
use mat::{Mat, DMat};
use std::sync::Arc;
use std::f32::consts::PI;
use std::thread::available_parallelism;
use std::fs::create_dir_all;
use image::RgbImage;
use clap::{Parser, ValueEnum};

/// Convert a value from `0.0 .. 1.0` to `0..255u8`
///
/// Any values outside this range are clamped to fit
fn to255(v : f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl From<&DMat<Color>> for RgbImage {
    fn from(mat : &DMat<Color>) -> RgbImage {
	let mut v = Vec::with_capacity(mat.size[0] * mat.size[1] * 3);

	for p in mat.data.iter() {
	    v.push(to255(p[0]));
	    v.push(to255(p[1]));
	    v.push(to255(p[2]));
	}
	
	RgbImage::from_vec(
	    mat.size[1] as u32,
	    mat.size[0] as u32,
	    v
	).unwrap()
    }
}

/// Apply a (low-intensity) blur to the [`Mat`] for a bloom effect
fn add_bloom(mat : &DMat<Color>, r : i32, c : f32) -> DMat<Color> {
    let [h, w] = mat.size;
    let mut res = DMat::new([h, w], Color::zero());

    // The factor for the pixel at the center
    let rem = 1.0 - (((2 * r + 1) as f32).powi(2) - 1.0) * c;
    
    for y in 0..h {
	for x in 0..w {
	    let mut s = Color::zero();
	    for dy in -r..=r {
		let ny = y as i32 + dy;
		if ny < 0 || ny >= h as i32 {
		    continue;
		}
		for dx in -r..=r {
		    let nx = x as i32 + dx;
		    if nx < 0 || nx >= w as i32 {
			continue;
		    }
		    if dy == 0 && dx == 0 {
			s += mat[y][x] * rem;
		    } else {
			s += mat[ny as usize][nx as usize] * c;
		    }
		}
	    }
	    res[y][x] = s;
	}
    }
    
    res
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Profile {
    High,
    HighHalf,
    Mid,
    Low,
    Single
}

#[derive(Parser)]
struct Args {
    #[arg(short, long, value_enum, default_value_t=Profile::Mid)]
    profile: Profile,

    #[arg(short, long, default_value_t=0)]
    threads: i32,

    #[arg(short, long)]
    animation: bool,

    #[arg(short, long, default_value_t=240)]
    frames: i32
}

fn main() {
    // Create the scene
    let main_cam = Camera {
	transform: Transform::translation(mat![0.0, 1.0, 0.0]),
	h_fov: PI / 2.0,
    };

    let mut scene : Scene = Default::default();

    scene.ior = Complex::from(1.0);
    
    scene.sky = Sky {
	color: mat![0.75, 0.9, 0.9] * 0.5,
	patches: vec![Patch {
	    direction: mat![-3.0, 2.5, -5.0].normalize(),
	    min_cos: (PI / 40.0).cos(),
	    color: mat![1.0, 1.0, 0.7] * 5.0
	}]
    };
    
    let gray = Arc::new(PointMaterial {
	color: mat![0.7, 0.7, 0.7],
	emissive: Color::zero(),
	roughness: 2.0,
	diffuse: 1.0,
	transmission: 0.0,
	ior: 1.457.into()
    });

    let plane_mesh = triangle_march(&vec![
	mat![-0.5, 0.0,  0.5],
	mat![ 0.5, 0.0,  0.5],
	mat![-0.5, 0.0, -0.5],
	mat![ 0.5, 0.0, -0.5],
    ]);

    let cube_mesh = triangle_march(&vec![
	mat![ 0.5, -0.5,  0.5],
	mat![-0.5, -0.5,  0.5],
	mat![ 0.5, -0.5, -0.5],
	mat![-0.5, -0.5, -0.5],
	mat![-0.5,  0.5, -0.5],
	mat![-0.5, -0.5,  0.5],
	mat![-0.5,  0.5,  0.5],
	mat![ 0.5, -0.5,  0.5],
	mat![ 0.5,  0.5,  0.5],
	mat![ 0.5, -0.5, -0.5],
	mat![ 0.5,  0.5, -0.5],
	mat![-0.5,  0.5, -0.5],
	mat![ 0.5,  0.5,  0.5],
	mat![-0.5,  0.5,  0.5],
    ]);

    let prism_mesh = triangle_march(&vec![
	mat![ 0.5, -0.5, -0.5],
	mat![-0.5,  0.5, -0.5],
	mat![ 0.5,  0.5, -0.5],
	mat![-0.5,  0.5,  0.5],
	mat![-0.5, -0.5,  0.5],
	mat![-0.5,  0.5, -0.5],
	mat![-0.5, -0.5, -0.5],
	mat![ 0.5, -0.5, -0.5],
	mat![-0.5, -0.5,  0.5],
	mat![ 0.5,  0.5, -0.5],
    ]);
    
    let silver_mirror = Arc::new(PointMaterial {
	color: mat![0.99, 0.99, 0.99],
	emissive: Color::zero(),
	roughness: 0.0,
	diffuse: 1.0,
	transmission: 0.0,
	ior: Complex {re: 0.135, im: 3.985}
    });
    
    let ground = Arc::new(SimpleMesh {
	material: gray,
	tris: plane_mesh.clone()
    });
    scene.objects.push(SceneObject::new(
	Transform::scale(mat![100.0, 1.0, 100.0]),
	ground
    ));

    let red = Arc::new(PointMaterial {
	color: mat![1.0, 0.3, 0.3],
	emissive: Color::zero(),
	roughness: 0.3,
	diffuse: 1.0,
	transmission: 0.0,
	ior: 1.55.into()
    });
    
    let cube = Arc::new(SimpleMesh {
	material: red.clone(),
	tris: cube_mesh
    });
    scene.objects.push(SceneObject::new(
	Transform::translation(mat![0.0, 0.9, -4.0])
	    * Transform::rotation_xyz(-PI / 4.0, PI / 4.0, -PI / 8.0),
	cube
    ));
    
    let sphere = Arc::new(Sphere {
	material: silver_mirror
    });
    
    scene.objects.push(SceneObject::new(
	Transform::translation(mat![-2.0, 1.5, -5.0]),
	sphere
    ));

    let glass = Arc::new(PointMaterial {
	color: mat![1.0, 1.0, 1.0],
	emissive: Color::zero(),
	roughness: 0.0,
	diffuse: 1.0,
	transmission: 0.95,
	ior: 1.5.into()
    });

    let prism = Arc::new(SimpleMesh {
	material: glass.clone(),
	tris: prism_mesh
    });

    scene.objects.push(SceneObject::new(
	Transform::translation(mat![0.3, 0.7, -2.5]) *
	    Transform::scale(mat![0.7, 0.7, 0.7]),
	prism
    ));
    
    // Parse arguments
    let args = Args::parse();

    let threads = if args.threads <= 0 {
	available_parallelism().map_or(4, |n| n.get() as i32)
    } else {args.threads};

    // Select the rendering profile
    let high = RenderProperties {
	resolution: [1920, 1080],
	passes: 15,
	bounces: 4,
	bounce_split: 4,
	threads,
	pixels_per_thread: 2500
    };

    let mut high_half = high.clone();
    high_half.resolution[0] /= 2;
    high_half.resolution[1] /= 2;
    
    let mid = RenderProperties {
	resolution: [640, 480],
	passes: 15,
	bounces: 3,
	bounce_split: 3,
	threads,
	pixels_per_thread: 2500
    };
    let low = RenderProperties {
	resolution: [640, 480],
	passes: 2,
	bounces: 2,
	bounce_split: 4,
	threads,
	pixels_per_thread: 2500
    };

    let single = RenderProperties {
	resolution: [640, 480],
	passes: 15,
	bounces: 3,
	bounce_split: 3,
	threads: 1,
	pixels_per_thread: 2500
    };
    
    let profile = match args.profile {
	Profile::High => &high,
	Profile::HighHalf => &high_half,
	Profile::Mid => &mid,
	Profile::Low => &low,
	Profile::Single => &single,
    };

    // Make sure that the image directory exists
    create_dir_all("img").unwrap();

    if args.animation {
	let prism_initial = scene.objects[3].transform;
	// Render an animation
	for f in 0..args.frames {
	    let p = (f as f32) / (args.frames as f32);
	    println!("Frame {}", f);

	    let cube_pos = mat![0.0, 0.9, -4.0];
	    
	    scene.objects[1].transform =
		Transform::translation(cube_pos) *
		Transform::rotation_xyz(0.0, p * 4.0 * PI, 0.0) *
		Transform::rotation_xyz(-PI / 4.0, PI / 4.0, -PI / 8.0);
	    
	    scene.objects[2].transform =
		Transform::translation(cube_pos) *
		Transform::rotation_xyz(0.0, p * 2.0 * PI, 0.0) *
		Transform::translation(mat![-2.0, 1.5, -5.0] - cube_pos);

	    scene.objects[3].transform =
		prism_initial *
		Transform::rotation_xyz(p * 2.0 * PI, 0.0, p * 2.0 * PI);
	    
	    scene.calc_cache();
	    let frame = main_cam.render(profile, &scene);
	    let frame = add_bloom(&frame, 3, 0.005);
	    let img = RgbImage::from(&frame);
	    img.save(format!("img/{:03}.png", f)).unwrap();
	}
    } else {
	// Render a single frame
	scene.calc_cache();
	let frame = main_cam.render(profile, &scene);
	let frame = add_bloom(&frame, 3, 0.005);
	let img = RgbImage::from(&frame);
	img.save("img/frame.png").unwrap();
    }
}
