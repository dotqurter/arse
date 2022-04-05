use winit:: { event_loop::{ControlFlow, EventLoop},
              window::{ WindowBuilder, Window },
              event::* };
use vulkano::{ instance::{ Instance, InstanceCreateInfo },
               device:: { physical::PhysicalDevice, physical::PhysicalDeviceType, DeviceExtensions, DeviceCreateInfo, QueueCreateInfo, Device },
               buffer::{ BufferUsage, CpuAccessibleBuffer, TypedBufferAccess },
               command_buffer::{ AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents },
               swapchain::{ Swapchain, SwapchainCreateInfo, SwapchainCreationError, acquire_next_image, AcquireError },
               image::{ ImageUsage, SwapchainImage, view::ImageView, ImageAccess },
               render_pass::{ Framebuffer, FramebufferCreateInfo, RenderPass, Subpass },
               pipeline::{ GraphicsPipeline, graphics::{ input_assembly::InputAssemblyState, vertex_input::BuffersDefinition, viewport::{ Viewport, ViewportState} } },
               sync::{ self, FlushError, GpuFuture },
               impl_vertex};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use vulkano::sync::now;

fn main() {
    //vulkan setup
    let req_ext = vulkano_win::required_extensions();
    let  dev_ext = DeviceExtensions {
        khr_swapchain: true, ..DeviceExtensions::none() };
    let vkinst = Instance::new(InstanceCreateInfo { enabled_extensions: req_ext, ..Default::default() })
        .expect("vkinst failed creation");
    
    //winit setup
    let event_loop = EventLoop::new();
    let builder = WindowBuilder::new();
    let window = builder.build_vk_surface(&event_loop, vkinst.clone()).unwrap();

    let (physical, queue_fam) = PhysicalDevice::enumerate(&vkinst)
        .filter(|&p| { p.supported_extensions().is_superset_of(&dev_ext) })
        .filter_map( |p|  {
            p.queue_families()
                .find(|&q| {
                    q.supports_graphics() && q.supports_surface(&window).unwrap_or(false)
                })
                .map(|q|  (p, q))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            }
        }).unwrap();
    
    let (dev, mut queues) = Device::new( physical, DeviceCreateInfo {
        enabled_extensions: physical.required_extensions().union(&dev_ext),
        queue_create_infos: vec![QueueCreateInfo::family(queue_fam)], ..Default::default() } )
        .expect("failed dev creation");
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_cap = physical.surface_capabilities(&window, Default::default())
            .unwrap();
        let image_format = Some(physical.surface_formats(&window, Default::default())
                                .unwrap()[0].0, );        
        Swapchain::new(dev.clone(), window.clone(), SwapchainCreateInfo {
            min_image_count: surface_cap.min_image_count,
            image_format,
            image_extent: window.window().inner_size().into(),
            image_usage: ImageUsage::color_attachment(),
            composite_alpha: surface_cap.supported_composite_alpha.iter().next().unwrap(), ..Default::default() }, ).unwrap()
    };

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex { position: [f32; 2], }
    impl_vertex!(Vertex, position);

    let vertices = [ Vertex { position: [-0.5, -0.25] }, Vertex { position: [0.0, 0.5] }, Vertex { position: [0.25, -0.1] },];
    let vertex_buffer = CpuAccessibleBuffer::from_iter(dev.clone(), BufferUsage::all(), false, vertices).unwrap();

    mod vs { //vertex shader
        vulkano_shaders::shader! { ty: "vertex",
        src: "#version 450

				layout(location = 0) in vec2 position;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}"
        }
    }
    mod fs {
        vulkano_shaders::shader!{ ty: "fragment",
        src: "#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}"
        }
    }
    let vs = vs::load(dev.clone()).unwrap();
    let fs = fs::load(dev.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!( dev.clone(),
                                                        attachments: { color: { load: Clear, store: Store, format: swapchain.image_format(), samples: 1,}},
                                                        pass: { color: [color], depth_stencil: {} }).unwrap();
    let pipeline = GraphicsPipeline::start().vertex_input_state(
        BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(dev.clone()).unwrap();

    let mut viewport = Viewport { origin: [0.0, 0.0], dimensions: [0.0, 0.0], depth_range: 0.0..1.0};
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(vulkano::sync::now(dev.clone()).boxed());
    
    //winit event loop.
    event_loop.run(move | event, _, control_flow |  {
        *control_flow = ControlFlow::Poll;
        //*control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { println!("Close button pressed."); *control_flow = ControlFlow::Exit },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => { recreate_swapchain = true; }
            Event::MainEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();
                if recreate_swapchain {
                    let (new_swapchain, new_images)  =
                        match swapchain.recreate(
                            SwapchainCreateInfo {
                                image_extent: window.window().inner_size().into(), ..swapchain.create_info() 
                            }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported {..}) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };
                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
                    recreate_swapchain = false;
                }
                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };
                if suboptimal { recreate_swapchain = true; }
                let clear_values = vec![ [0.0, 0.0, 1.0, 1.0].into() ];

                let mut builder = AutoCommandBufferBuilder::primary(dev.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
                builder.begin_render_pass(framebuffers[image_num].clone(), SubpassContents::Inline, clear_values).unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap()
                    .end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();
                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num).then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(vulkano::sync::now(dev.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(vulkano::sync::now(dev.clone()).boxed());
                    }
                }
            }
            _ => ()
        }
    });
}

 /// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
