Split Raster Chunk into 6
cull chunks that are outside of view

Make a UI

Make a spell
Misc:
    - everything refactor
    - scramble draw order dogfood!!!
    - Consume memory ordering bikeshedding?
    - Custom Linear Algebra library?
    - Put cache directly in renderer

pat | #c | scale| side  | total
3x3 | 9  | 1.0  | 512   | 1536
4r4 | 12 | 1.5  | 768   | 3072 
4r4 | 12 | 3.0  | 1536  | 6144 
4r4 | 12 | 6.0  | 3072  | 12288
4r4 | 12 | 12.0 | 6144  | 24576
4r4 | 12 | 12.0 | 12288 | 49152
4r4 | 12 | 12.0 | 24576 | 98304
-------------------------------
Total Chunks: 81
Rendered area: 98304^2 x 512 

greedy meshing
draw indriet
face level culling

jemealloc

create an mvp

make a really good readme



            //     for pass_type in PassStage::iter()
            //     {
            //         let mut render_pass: GenericPass = match pass_type
            //         {
            //             PassStage::VoxelDiscovery =>
            //             {
            //                 GenericPass::Render(encoder.begin_render_pass(
            //                     &wgpu::RenderPassDescriptor {
            //                         label:                    Some("Voxel Discovery
            // Pass"),                         color_attachments:        &[Some(
            //                             wgpu::RenderPassColorAttachment {
            //                                 view:           voxel_discovery_view,
            //                                 resolve_target: None,
            //                                 ops:            wgpu::Operations {
            //                                     load:  wgpu::LoadOp::Clear(wgpu::Color {
            //                                         r: 0.0,
            //                                         g: 0.0,
            //                                         b: 0.0,
            //                                         a: 0.0
            //                                     }),
            //                                     store: wgpu::StoreOp::Store
            //                                 }
            //                             }
            //                         )],
            //                         depth_stencil_attachment: Some(
            //                             wgpu::RenderPassDepthStencilAttachment {
            //                                 view:        depth_view,
            //                                 depth_ops:   Some(wgpu::Operations {
            //                                     load:  wgpu::LoadOp::Clear(1.0),
            //                                     store: wgpu::StoreOp::Store
            //                                 }),
            //                                 stencil_ops: None
            //                             }
            //                         ),
            //                         occlusion_query_set:      None,
            //                         timestamp_writes:         None
            //                     }
            //                 ))
            //             }
            //             PassStage::PostVoxelDiscoveryCompute =>
            //             {
            //                 GenericPass::Compute(encoder.begin_compute_pass(
            //                     &wgpu::ComputePassDescriptor {
            //                         label:            Some("Post Voxel Discovery
            // Compute"),                         timestamp_writes: None
            //                     }
            //                 ))
            //             }
            //             PassStage::VoxelColorTransfer =>
            //             {
            //                 GenericPass::Render(encoder.begin_render_pass(
            //                     &wgpu::RenderPassDescriptor {
            //                         label:                    Some("Voxel Color Transfer
            // Pass"),                         color_attachments:        &[Some(
            //                             wgpu::RenderPassColorAttachment {
            //                                 view:           &screen_texture_view,
            //                                 resolve_target: None,
            //                                 ops:            wgpu::Operations {
            //                                     load:  wgpu::LoadOp::Clear(wgpu::Color {
            //                                         r: 0.1,
            //                                         g: 0.2,
            //                                         b: 0.3,
            //                                         a: 1.0
            //                                     }),
            //                                     store: wgpu::StoreOp::Store
            //                                 }
            //                             }
            //                         )],
            //                         depth_stencil_attachment: None,
            //                         occlusion_query_set:      None,
            //                         timestamp_writes:         None
            //                     }
            //                 ))
            //             }
            //             PassStage::SimpleColor =>
            //             {
            //                 GenericPass::Render(encoder.begin_render_pass(
            //                     &wgpu::RenderPassDescriptor {
            //                         label:                    Some("Simple Color Pass"),
            //                         color_attachments:        &[Some(
            //                             wgpu::RenderPassColorAttachment {
            //                                 view:           &screen_texture_view,
            //                                 resolve_target: None,
            //                                 ops:            wgpu::Operations {
            //                                     load:  wgpu::LoadOp::Load,
            //                                     store: wgpu::StoreOp::Store
            //                                 }
            //                             }
            //                         )],
            //                         depth_stencil_attachment: Some(
            //                             wgpu::RenderPassDepthStencilAttachment {
            //                                 view:        depth_view,
            //                                 depth_ops:   Some(wgpu::Operations {
            //                                     load:  wgpu::LoadOp::Load,
            //                                     store: wgpu::StoreOp::Store
            //                                 }),
            //                                 stencil_ops: None
            //                             }
            //                         ),
            //                         occlusion_query_set:      None,
            //                         timestamp_writes:         None
            //                     }
            //                 ))
            //             }
            //             PassStage::MenuRender =>
            //             {
            //                 GenericPass::Render(encoder.begin_render_pass(
            //                     &wgpu::RenderPassDescriptor {
            //                         label:                    None,
            //                         color_attachments:        &[Some(
            //                             wgpu::RenderPassColorAttachment {
            //                                 view:           &screen_texture_view,
            //                                 resolve_target: None,
            //                                 ops:            wgpu::Operations {
            //                                     load:  wgpu::LoadOp::Load,
            //                                     store: wgpu::StoreOp::Store
            //                                 }
            //                             }
            //                         )],
            //                         depth_stencil_attachment: None,
            //                         timestamp_writes:         None,
            //                         occlusion_query_set:      None
            //                     }
            //                 ))
            //             }
            //         };

            //