use crate::{indexed_name_to_ident, wgsl::buffer_binding_type, CreateModuleError};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::collections::BTreeMap;
use syn::{Ident, Index};

pub struct GroupData<'a> {
    pub bindings: Vec<GroupBinding<'a>>,
}

pub struct GroupBinding<'a> {
    pub name: Option<String>,
    pub binding_index: u32,
    pub binding_type: &'a naga::Type,
    pub address_space: naga::AddressSpace,
}

// TODO: Take an iterator instead?
pub fn bind_groups_module(
    bind_group_data: &BTreeMap<u32, GroupData>,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let bind_groups: Vec<_> = bind_group_data
        .iter()
        .map(|(group_no, group)| {
            let group_name = indexed_name_to_ident("BindGroup", *group_no);

            let layout = bind_group_layout(*group_no, group);
            let layout_descriptor = bind_group_layout_descriptor(*group_no, group, shader_stages);
            let group_impl = bind_group(*group_no, group, shader_stages);

            quote! {
                #[derive(Debug, Clone)]
                pub struct #group_name(BindGroup);
                #layout
                #layout_descriptor
                #group_impl
            }
        })
        .collect();

    let bind_group_fields: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group_name = indexed_name_to_ident("BindGroup", *group_no);
            let field = indexed_name_to_ident("bind_group", *group_no);
            quote!(pub #field: &'a #group_name)
        })
        .collect();

    // TODO: Support compute shader with vertex/fragment in the same module?
    let is_compute = shader_stages == wgpu::ShaderStages::COMPUTE;
    let render_pass = if is_compute {
        quote!(ComputePass<'a>)
    } else {
        quote!(RenderPass<'a>)
    };

    let group_parameters: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("bind_group", *group_no);
            let group_type = indexed_name_to_ident("BindGroup", *group_no);
            quote!(#group: &'a bind_groups::#group_type)
        })
        .collect();

    // The set function for each bind group already sets the index.
    let set_groups: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("bind_group", *group_no);
            quote!(#group.set(pass);)
        })
        .collect();

    let set_bind_groups = quote! {
        pub fn set_bind_groups<'a>(
            pass: &mut #render_pass,
            #(#group_parameters),*
        ) {
            #(#set_groups)*
        }
    };

    if bind_groups.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        // Create a module to avoid name conflicts with user structs.
        quote! {
            pub mod bind_groups { 
                use bevy::{ecs::system::Resource, render::{render_resource::*, renderer::RenderDevice}};

                #(#bind_groups)*

                pub struct BindGroups<'a> {
                #[derive(Debug, Copy, Clone, Resource)]
                    #(#bind_group_fields),*
                }

                impl<'a> BindGroups<'a> {
                    pub fn set(&self, pass: &mut #render_pass) {
                        #(self.#set_groups)*
                    }
                }
            }
            #set_bind_groups
        }
    }
}

fn bind_group_layout(group_no: u32, group: &GroupData) -> TokenStream {
    let fields: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| {
            let field_name = Ident::new(binding.name.as_ref().unwrap(), Span::call_site());
            // TODO: Support more types.
            let field_type = match binding.binding_type.inner {
                naga::TypeInner::Struct { .. } => quote!(BufferBinding<'a>),
                naga::TypeInner::Image { .. } => quote!(&'a TextureView),
                naga::TypeInner::Sampler { .. } => quote!(&'a Sampler),
                naga::TypeInner::Array { .. } => quote!(BufferBinding<'a>),
                _ => panic!("Unsupported type for binding fields."),
            };
            quote!(pub #field_name: #field_type)
        })
        .collect();

    let name = indexed_name_to_ident("BindGroupLayout", group_no);
    quote! {
        #[derive(Debug)]
        pub struct #name<'a> {
            #(#fields),*
        }
    }
}

// fn struct_buffer_layout(group_no: u32, group: &GroupData) -> TokenStream {
//     let fields: Vec<_> = group
//         .bindings
//         .iter()
//         .map(|binding| {
//             let field_name = Ident::new(binding.name.as_ref().unwrap(), Span::call_site());
//             // TODO: Support more types.
//             let field_type = match binding.binding_type.inner {
//                 naga::TypeInner::Struct { .. } => quote!(BufferBinding<'a>),
//                 naga::TypeInner::Image { .. } => quote!(&'a TextureView),
//                 naga::TypeInner::Sampler { .. } => quote!(&'a Sampler),
//                 naga::TypeInner::Array { .. } => quote!(BufferBinding<'a>),
//                 _ => panic!("Unsupported type for binding fields."),
//             };
//             quote!(pub #field_name: #field_type)
//         })
//         .collect();

//     let name = indexed_name_to_ident("Buffers", group_no);
//     quote! {
//         #[derive(Debug, Resource)]
//         pub struct #name<'a> {
//             #(#fields),*
//         }
//     }
    
// }

fn bind_group_layout_descriptor(
    group_no: u32,
    group: &GroupData,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let entries: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| bind_group_layout_entry(binding, shader_stages))
        .collect();

    let name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", group_no);
    quote! {
        pub const #name: &[BindGroupLayoutEntry] = &[
                #(#entries),*
            ];
    }
}

fn bind_group_layout_entry(
    binding: &GroupBinding,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    // TODO: Assume storage is only used for compute?
    // TODO: Support just vertex or fragment?
    // TODO: Visible from all stages?
    let stages = match shader_stages {
        wgpu::ShaderStages::VERTEX_FRAGMENT => quote!(ShaderStages::VERTEX_FRAGMENT),
        wgpu::ShaderStages::COMPUTE => quote!(ShaderStages::COMPUTE),
        wgpu::ShaderStages::VERTEX => quote!(ShaderStages::VERTEX),
        wgpu::ShaderStages::FRAGMENT => quote!(ShaderStages::FRAGMENT),
        _ => todo!(),
    };

    let binding_index = Index::from(binding.binding_index as usize);
    // TODO: Support more types.
    let binding_type = match binding.binding_type.inner {
        naga::TypeInner::Struct { .. } => {
            let buffer_binding_type = buffer_binding_type(binding.address_space);

            quote!(BindingType::Buffer {
                ty: #buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            })
        }
        naga::TypeInner::Array { .. } => {
            let buffer_binding_type = buffer_binding_type(binding.address_space);

            quote!(BindingType::Buffer {
                ty: #buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            })
        }
        naga::TypeInner::Image { dim, class, .. } => {
            let view_dim = match dim {
                naga::ImageDimension::D1 => quote!(TextureViewDimension::D1),
                naga::ImageDimension::D2 => quote!(TextureViewDimension::D2),
                naga::ImageDimension::D3 => quote!(TextureViewDimension::D3),
                naga::ImageDimension::Cube => quote!(TextureViewDimension::Cube),
            };

            match class {
                naga::ImageClass::Sampled { kind, multi } => {
                    let sample_type = match kind {
                        naga::ScalarKind::Sint => quote!(TextureSampleType::Sint),
                        naga::ScalarKind::Uint => quote!(TextureSampleType::Uint),
                        naga::ScalarKind::Float => {
                            // TODO: Don't assume all textures are filterable.
                            quote!(TextureSampleType::Float { filterable: true })
                        }
                        _ => todo!(),
                    };
                    quote!(BindingType::Texture {
                        sample_type: #sample_type,
                        view_dimension: #view_dim,
                        multisampled: #multi,
                    })
                }
                naga::ImageClass::Depth { multi } => {
                    quote!(BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: #view_dim,
                        multisampled: #multi,
                    })
                }
                naga::ImageClass::Storage { format, access } => {
                    // TODO: Will the debug implementation always work with the macro?
                    // Assume texture format variants are the same as storage formats.
                    let format = syn::Ident::new(&format!("{format:?}"), Span::call_site());
                    let storage_access = storage_access(access);

                    quote!(BindingType::StorageTexture {
                        access: #storage_access,
                        format: TextureFormat::#format,
                        view_dimension: #view_dim,
                    })
                }
            }
        }
        naga::TypeInner::Sampler { comparison } => {
            let sampler_type = if comparison {
                quote!(SamplerBindingType::Comparison)
            } else {
                quote!(SamplerBindingType::Filtering)
            };
            quote!(BindingType::Sampler(#sampler_type))
        }
        // TODO: Better error handling.
        _ => panic!("Failed to generate BindingType."),
    };

    quote! {
        BindGroupLayoutEntry {
            binding: #binding_index,
            visibility: #stages,
            ty: #binding_type,
            count: None,
        }
    }
}

fn storage_access(access: naga::StorageAccess) -> TokenStream {
    let is_read = access.contains(naga::StorageAccess::LOAD);
    let is_write = access.contains(naga::StorageAccess::STORE);
    match (is_read, is_write) {
        (true, true) => quote!(StorageTextureAccess::ReadWrite),
        (true, false) => quote!(StorageTextureAccess::ReadOnly),
        (false, true) => quote!(StorageTextureAccess::WriteOnly),
        _ => todo!(), // shouldn't be possible
    }
}

fn bind_group(group_no: u32, group: &GroupData, shader_stages: wgpu::ShaderStages) -> TokenStream {
    let entries: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| {
            let binding_index = Index::from(binding.binding_index as usize);
            let binding_name = Ident::new(binding.name.as_ref().unwrap(), Span::call_site());
            let resource_type = match binding.binding_type.inner {
                naga::TypeInner::Struct { .. } => {
                    quote!(BindingResource::Buffer(bindings.#binding_name))
                }
                naga::TypeInner::Array { .. } => {
                    quote!(BindingResource::Buffer(bindings.#binding_name))
                }
                naga::TypeInner::Image { .. } => {
                    quote!(BindingResource::TextureView(bindings.#binding_name))
                }
                naga::TypeInner::Sampler { .. } => {
                    quote!(BindingResource::Sampler(bindings.#binding_name))
                }
                // TODO: Better error handling.
                _ => panic!("Failed to generate BindingType."),
            };

            quote! {
                BindGroupEntry {
                    binding: #binding_index,
                    resource: #resource_type,
                }
            }
        })
        .collect();

    // TODO: Support compute shader with vertex/fragment in the same module?
    let is_compute = shader_stages == wgpu::ShaderStages::COMPUTE;

    let render_pass = if is_compute {
        quote!(ComputePass<'a>)
    } else {
        quote!(RenderPass<'a>)
    };

    let bind_group_name = indexed_name_to_ident("BindGroup", group_no);
    let bind_group_layout_name = indexed_name_to_ident("BindGroupLayout", group_no);
    let layout_descriptor_name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", group_no);

    let bind_group_name_str = format!("BindGroup{group_no}");
    let bind_group_layout_name_str = format!("BindGroupLayout{group_no}");

    let group_no = Index::from(group_no as usize);
    // let entries: Vec<_> = group
    //     .bindings
    //     .iter()
    //     .map(|binding| bind_group_layout_entry(binding, shader_stages))
    //     .collect();
    quote! {
        impl #bind_group_name {
            pub fn get_bind_group_layout_entry() -> Vec<BindGroupLayoutEntry> {
                #layout_descriptor_name.to_vec()
            }
            pub fn get_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
                render_device.create_bind_group_layout(#bind_group_layout_name_str, &#layout_descriptor_name)
            }

            pub fn from_bindings(render_device: &RenderDevice, bindings: #bind_group_layout_name) -> Self {
                let bind_group_layout = render_device.create_bind_group_layout(#bind_group_layout_name_str, &#layout_descriptor_name);
                let bind_group = render_device.create_bind_group(
                    #bind_group_name_str,
                    &bind_group_layout,
                    &[
                        #(#entries),*
                    ],
                );
                Self(bind_group)
            }

            pub fn set<'a>(&'a self, render_pass: &mut #render_pass) {
                render_pass.set_bind_group(#group_no, &self.0, &[]);
            }
        }
    }
}

pub fn get_bind_group_data(
    module: &naga::Module,
) -> Result<BTreeMap<u32, GroupData>, CreateModuleError> {
    // Use a BTree to sort type and field names by group index.
    // This isn't strictly necessary but makes the generated code cleaner.
    let mut groups = BTreeMap::new();

    for global_handle in module.global_variables.iter() {
        let global = &module.global_variables[global_handle.0];
        if let Some(binding) = &global.binding {
            let group = groups.entry(binding.group).or_insert(GroupData {
                bindings: Vec::new(),
            });
            let binding_type = &module.types[module.global_variables[global_handle.0].ty];

            let group_binding = GroupBinding {
                name: global.name.clone(),
                binding_index: binding.binding,
                binding_type,
                address_space: global.space,
            };
            // Repeated bindings will probably cause a compile error.
            // We'll still check for it here just in case.
            if group
                .bindings
                .iter()
                .any(|g| g.binding_index == binding.binding)
            {
                return Err(CreateModuleError::DuplicateBinding {
                    binding: binding.binding,
                });
            }
            group.bindings.push(group_binding);
        }
    }

    // wgpu expects bind groups to be consecutive starting from 0.
    if groups.keys().map(|i| *i as usize).eq(0..groups.len()) {
        Ok(groups)
    } else {
        Err(CreateModuleError::NonConsecutiveBindGroups)
    }
}
