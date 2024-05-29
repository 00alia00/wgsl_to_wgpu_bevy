//! # wgsl_to_wgpu
//! wgsl_to_wgpu is a library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Getting Started
//! The [create_shader_module] function is intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! ```rust no_run
//! // build.rs
//! use wgsl_to_wgpu::{create_shader_module, MatrixVectorTypes, WriteOptions};
//!
//! fn main() {
//!     println!("cargo:rerun-if-changed=src/shader.wgsl");
//!
//!     // Read the shader source file.
//!     let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();
//!
//!     // Configure the output based on the dependencies for the project.
//!     let options = WriteOptions {
//!         derive_bytemuck_vertex: true,
//!         derive_encase_host_shareable: true,
//!         matrix_vector_types: MatrixVectorTypes::Glam,
//!         ..Default::default()
//!     };
//!
//!     // Generate the bindings.
//!     let text = create_shader_module(&wgsl_source, "shader.wgsl", options).unwrap();
//!     std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
//! }
//! ```

extern crate wgpu_types as wgpu;

use bindgroup::{bind_groups_module, get_bind_group_data};
use consts::pipeline_overridable_constants;
use entry::{entry_point_constants, fragment_states, vertex_states, vertex_struct_methods};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Index};
use thiserror::Error;

mod bindgroup;
mod consts;
mod entry;
mod structs;
mod wgsl;

/// Errors while generating Rust source for a WGSl shader module.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum CreateModuleError {
    /// Bind group sets must be consecutive and start from 0.
    /// See `bind_group_layouts` for
    /// [PipelineLayoutDescriptor](https://docs.rs/wgpu/latest/wgpu/struct.PipelineLayoutDescriptor.html#).
    #[error("bind groups are non-consecutive or do not start from 0")]
    NonConsecutiveBindGroups,

    /// Each binding resource must be associated with exactly one binding index.
    #[error("duplicate binding found with index `{binding}`")]
    DuplicateBinding { binding: u32 },
}

/// Options for configuring the generated bindings to work with additional dependencies.
/// Use [WriteOptions::default] for only requiring WGPU itself.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct WriteOptions {
    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for WGSL vertex input structs when `true`.
    pub derive_bytemuck_vertex: bool,

    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for user defined WGSL structs for host-shareable types (uniform and storage buffers) when `true`.
    ///
    /// This will generate compile time assertions to check that the memory layout
    /// of structs and struct fields matches what is expected by WGSL.
    /// This does not account for all layout and alignment rules like storage buffer offset alignment.
    ///
    /// Most applications should instead handle these requirements more reliably at runtime using encase.
    pub derive_bytemuck_host_shareable: bool,

    /// Derive [encase::ShaderType](https://docs.rs/encase/latest/encase/trait.ShaderType.html#)
    /// for user defined WGSL structs for host-shareable types (uniform and storage buffers) when `true`.
    /// Use [MatrixVectorTypes::Glam] for best compatibility.
    pub derive_encase_host_shareable: bool,

    /// Derive [serde::Serialize](https://docs.rs/serde/1.0.159/serde/trait.Serialize.html)
    /// and [serde::Deserialize](https://docs.rs/serde/1.0.159/serde/trait.Deserialize.html)
    /// for user defined WGSL structs when `true`.
    pub derive_serde: bool,

    /// The format to use for matrix and vector types.
    pub matrix_vector_types: MatrixVectorTypes,
}

/// The format to use for matrix and vector types.
/// Note that the generated types for the same WGSL type may differ in size or alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixVectorTypes {
    /// Rust types like `[f32; 4]` or `[[f32; 4]; 4]`.
    Rust,

    /// `glam` types like `glam::Vec4` or `glam::Mat4`.
    /// Types not representable by `glam` like `mat2x3<f32>` will use the output from [MatrixVectorTypes::Rust].
    Glam,

    /// `nalgebra` types like `nalgebra::SVector<f64, 4>` or `nalgebra::SMatrix<f32, 2, 3>`.
    Nalgebra,
}

impl Default for MatrixVectorTypes {
    fn default() -> Self {
        Self::Rust
    }
}

/// Generates a Rust module for a WGSL shader included via [include_str].
///
/// The `wgsl_include_path` should be a valid input to [include_str] in the generated file's location.
/// The included contents should be identical to `wgsl_source`.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/**
```rust no_run
// build.rs
fn main() {
    println!("cargo:rerun-if-changed=src/shader.wgsl");

    // Read the shader source file.
    let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();

    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions {
        derive_bytemuck_vertex: true,
        derive_encase_host_shareable: true,
        matrix_vector_types: wgsl_to_wgpu::MatrixVectorTypes::Glam,
        ..Default::default()
    };

    // Generate the bindings.
    let text = wgsl_to_wgpu::create_shader_module(&wgsl_source, "shader.wgsl", options).unwrap();
    std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
}
```
 */
pub fn create_shader_module(
    wgsl_source: &str,
    wgsl_include_path: &str,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    create_shader_module_inner(wgsl_source, Some(wgsl_include_path), options)
}

// TODO: Show how to convert a naga module back to wgsl.
/// Generates a Rust module for a WGSL shader embedded as a string literal.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/// The source string does not need to be from an actual file on disk.
/// This allows applying build time operations like preprocessor defines.
/**
```rust no_run
// build.rs
# fn generate_wgsl_source_string() -> String { String::new() }
fn main() {
    // Generate the shader at build time.
    let wgsl_source = generate_wgsl_source_string();

    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions {
        derive_bytemuck_vertex: true,
        derive_encase_host_shareable: true,
        matrix_vector_types: wgsl_to_wgpu::MatrixVectorTypes::Glam,
        ..Default::default()
    };

    // Generate the bindings.
    let text = wgsl_to_wgpu::create_shader_module_embedded(&wgsl_source, options).unwrap();
    std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
}
```
 */
pub fn create_shader_module_embedded(
    wgsl_source: &str,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    create_shader_module_inner(wgsl_source, None, options)
}

fn create_shader_module_inner(
    wgsl_source: &str,
    wgsl_include_path: Option<&str>,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap();

    let bind_group_data = get_bind_group_data(&module)?;
    let shader_stages = wgsl::shader_stages(&module);

    // Write all the structs, including uniforms and entry function inputs.
    let structs = structs::structs(&module, options);
    let consts = consts::consts(&module);
    let bind_groups_module = bind_groups_module(&bind_group_data, shader_stages);
    let vertex_module = vertex_struct_methods(&module);
    let compute_module = compute_module(&module);
    let entry_point_constants = entry_point_constants(&module);
    let vertex_states = vertex_states(&module);
    let fragment_states = fragment_states(&module);

    let create_shader_module = quote! {
        use bevy::prelude::*;

        pub fn create_shader_module(world: &World) -> Handle<Shader> {
            world.resource::<AssetServer>().load(#wgsl_include_path)            
        }
    };

    let bind_group_layouts: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("BindGroup", *group_no);
            quote!(bind_groups::#group::get_bind_group_layout(render_device))
        })
        .collect();


    let create_pipeline_layout = quote! {

        pub fn create_pipeline_layout(render_device: &RenderDevice) -> PipelineLayout {
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    #(&#bind_group_layouts),*
                ],
                push_constant_ranges: &[],
            })
        }
    };

    let bind_group_layouts: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("BindGroup", *group_no);
            quote!(bind_groups::#group::get_bind_group_layout_entry())
        })
        .collect();

    let create_bindgroup_layout = quote! {
        pub fn create_bindgroup_layout(render_device: &RenderDevice) -> Vec<BindGroupLayout> {
            let groups = vec![
                #(#bind_group_layouts),*
            ]; 
            let mut entries = Vec::<BindGroupLayout>::new();
            for group in &groups {
                let layout = render_device.create_bind_group_layout("BindGroupLayout", group);
                entries.push(layout);
            }
            entries
        }
    };
    let override_constants = pipeline_overridable_constants(&module);

    let output = quote! {
        #(#structs)*
        #(#consts)*
        #override_constants
        #bind_groups_module
        #vertex_module
        #compute_module
        #entry_point_constants
        #vertex_states
        #fragment_states
        #create_shader_module
        #create_pipeline_layout
        #create_bindgroup_layout
    };
    Ok(pretty_print(&output))
}

fn pretty_print(tokens: &TokenStream) -> String {
    let file = syn::parse_file(&tokens.to_string()).unwrap();
    prettyplease::unparse(&file)
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{name}{index}"), Span::call_site())
}

fn compute_module(module: &naga::Module) -> TokenStream {
    let entry_points: Vec<_> = module
        .entry_points
        .iter()
        .filter_map(|e| {
            if e.stage == naga::ShaderStage::Compute {
                let workgroup_size_constant = workgroup_size(e);
                let create_pipeline = create_compute_pipeline(e);

                Some(quote! {
                    #workgroup_size_constant
                    #create_pipeline
                })
            } else {
                None
            }
        })
        .collect();

    if entry_points.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod compute {
                use bevy::render::{render_resource::*, renderer::RenderDevice};
                use bevy::prelude::*;
                use std::borrow::Cow;
                
                #(#entry_points)*
            }
        }
    }
}

fn create_compute_pipeline(e: &naga::EntryPoint) -> TokenStream {
    // Compute pipeline creation has few parameters and can be generated.
    let pipeline_name = Ident::new(&format!("create_{}_pipeline", e.name.to_lowercase()), Span::call_site());
    let entry_point = &e.name;
    // TODO: Include a user supplied module name in the label?
    let label = format!("Compute Pipeline {}", e.name);
    quote! {
        pub fn #pipeline_name(world: &mut World) -> CachedComputePipelineId {
            let render_device = world.resource::<RenderDevice>();
            let pipeline_cache = world.resource::<PipelineCache>();
    
            let shader = super::create_shader_module(world);
            let layout = super::create_bindgroup_layout(render_device);

            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::from(#label)),
                layout: layout,
                push_constant_ranges: vec![],    
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from(#entry_point),
            })
        }
    }
}

fn workgroup_size(e: &naga::EntryPoint) -> TokenStream {
    // Use Index to avoid specifying the type on literals.
    let name = Ident::new(
        &format!("{}_WORKGROUP_SIZE", e.name.to_uppercase()),
        Span::call_site(),
    );
    let [x, y, z] = e.workgroup_size.map(|s| Index::from(s as usize));
    quote!(pub const #name: [u32; 3] = [#x, #y, #z];)
}

// Tokenstreams can't be compared directly using PartialEq.
// Use pretty_print to normalize the formatting and compare strings.
// Use a colored diff output to make differences easier to see.
#[cfg(test)]
#[macro_export]
macro_rules! assert_tokens_eq {
    ($a:expr, $b:expr) => {
        pretty_assertions::assert_eq!(crate::pretty_print(&$a), crate::pretty_print(&$b))
    };
}
