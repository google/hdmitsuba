# hdMitsuba User Guide: Supported Features & Render Settings

This document provides an overview of the USD features supported by `hdMitsuba`, custom render settings, and how to author USD scenes for rendering with Mitsuba 3. Most features are covered by unit tests, so the `test_assets` folder is also a good reference.

## 1. Supported USD Features

`hdMitsuba` translates many standard USD concepts into Mitsuba equivalents. We are continuously expanding the supported feature set, currently we support:

### Geometry & Animation
*   **Meshes**: Flat meshes, subdivision surfaces (Catmull-Clark), and deforming geometry (such as skeletal animation). Subdivision surfaces are evaluated at a fixed subdivision level before rendering (no on-the-fly or adaptive tessellation). The subdivision level can be overridden per mesh by authoring the `primvars:mitsuba:subdivision_level` (int) primvar. Creases, corners, and hole faces (`holeIndices`) are supported, the latter also on meshes that are not subdivided.
*   **Curves**: Linear and cubic (`bspline` and `catmullRom`, with `nonperiodic` and `pinned` wrap modes) `BasisCurves` (translated to Mitsuba curve shapes). USD supports additional curve types (e.g., Bezier) which Mitsuba itself currently does not support.
*   **Implicit Surfaces**: Spheres and cylinders are rendered as exact native Mitsuba shapes (cylinders as a `cylinder` with two `disk` caps), including under non-uniform scaling, which is handled by wrapping them in a Mitsuba instance. Since Mitsuba has no tapered cylinder, a cylinder with differing `radiusTop` and `radiusBottom` is rendered with its bottom radius. Displacement is not supported on these shapes. Cubes, planes, cones, and capsules are tessellated. Set the environment variable `HDMITSUBA_TESSELLATE_IMPLICIT_SURFACES=1` to tessellate spheres and cylinders as well.
*   **Instancing**: Full support for USD point instancers and instanced geometry references. Note that Mitsuba's instances cannot be light sources currently.
*   **Displacement Mapping**: Fully supports displacement mapping. The delegate evaluates displacement textures connected to `mitsuba:displacement` or standard USD displacement terminals, either directly or through a `UsdPreviewSurface`'s `inputs:displacement`. Scalar displacement values are used as-is as offsets along the vertex normal (there is no implicit 0.5 mid-level); remap them with `UsdUVTexture` `inputs:scale` and `inputs:bias`.

### Shading & Materials
*   **UsdPreviewSurface**: Various USD preview surface parameters are mapped to Mitsuba's `principled` BSDF:
    *   `diffuseColor` (Base color)
    *   `roughness` (Specular roughness)
    *   `metallic` (Metalness)
    *   `specular` (Specular reflectance)
    *   `clearcoat` & `clearcoatRoughness`
    *   `opacity` (Transmission)
    *   `ior` (Index of refraction)
    *   `normal` (Normal mapping; expects tangent-space normals in `[-1, 1]` as per the [UsdPreviewSurface spec](https://openusd.org/release/spec_usdpreviewsurface.html#preview-surface), i.e. 8-bit textures need `scale = (2, 2, 2, 1)`, `bias = (-1, -1, -1, 0)` and `sourceColorSpace = "raw"`)
*   **UsdUVTexture**: `inputs:scale` and `inputs:bias` are applied (`texture * scale + bias`), including to `inputs:fallback`. Scaled or biased textures are wrapped in a Mitsuba `math` texture, which does not support bump mapping (`mitsuba_bumpmap`), emissive textures with the `ptracer` integrator, or per-channel scale/bias on color inputs in spectral variants.
*   **Native Mitsuba BSDFs & Textures**: We support embedding native Mitsuba BSDFs and textures as custom USD shader prims. This allows accessing all of Mitsuba's built-in BSDFs. Examples for this are provided in `test_assets/materials`. Note that USD natively allows specifying both a generic preview surface and a renderer-specific material. hdMitsuba will automatically use the shader that is connected to `outputs:mitsuba:surface`, if available. 
*   **Primvar Readers**: `UsdPrimvarReader_*` shaders (`float`, `int`, `float2`, `float3`, `float4`, `normal`, `point`, and `vector`) connected to UsdPreviewSurface inputs or native Mitsuba shaders read mesh primvars of any interpolation as Mitsuba `mesh_attribute` textures. They return the authored primvar values, in object space. Their `fallback` input is not supported (missing primvars read as zero), and they are not supported in displacement and volume shaders.
*   **Fallback Shading (displayColor)**: If a mesh has no bound material, the delegate automatically maps the USD `displayColor` attribute to a fallback `diffuse` BSDF with its `reflectance` bound to the color.

### Lights & Camera

*   **Light Types**: Sphere lights, point lights, spot lights, rectangle lights,
    disk lights, cylinder lights, distant lights, and dome lights (for HDR
    environment map backgrounds). Light shaping (`UsdLuxShapingAPI`
    `inputs:shaping:*`, including Type C IES profiles, cone angle/softness, and
    focus) is supported on sphere, point, rectangle, disk, cylinder, and mesh
    lights. As OpenUSD doesn't define the IES profile orientation, it follows
    hdEmbree: `theta = 0` along the light's -Z axis (its emission direction),
    and `phi = 0` along its -X axis, increasing towards -Y. Standard USDLux area
    lights (`SphereLight` when not point, `RectLight`, `DiskLight`,
    `CylinderLight`) default to `"secondary"` ray visibility (invisible to
    primary camera rays while still illuminating the scene and reflecting in
    secondary rays). Ray visibility on area lights and dome lights can be
    controlled by setting the `mitsuba:visibility` (string or token) attribute
    (`"all"`, `"primary"`, `"secondary"`, `"hidden"`). Supported USD lights and
    geometry prims (`Mesh`, `Sphere`, `Cylinder`, `Cube`, and `Plane`) can also
    customize or attach native Mitsuba emitters and shape parameters via
    `mitsuba:emitter:*` and `mitsuba:shape:*` attributes.
*   **Cameras & Custom Sensors**:
    *   Supports standard perspective cameras with horizontal/vertical aperture
        offsets, clipping ranges, and thin-lens depth of field (`thinlens`)
        whenever both `fStop` and `focusDistance` are positive
        (`aperture_radius = 0.5 * focalLength_scene_units / fStop`).
    *   **Custom Sensor Specs (`mitsuba:sensor:*`)**: Customize the Mitsuba
        sensor plugin and any of its parameters or nested child plugins on a
        `Camera` prim (e.g., `mitsuba:sensor:type = "thinlens"` with
        `mitsuba:sensor:aperture_radius` and `mitsuba:sensor:focus_distance`, or
        `orthographic`, `radiancemeter`, `irradiancemeter`, `distant`).
        Authoring `mitsuba:sensor:sampler:*` configures a standalone per-camera
        sampler override (defaulting `type` to `"independent"` and
        `sample_count` to `4` when omitted).
    *   **Film & Pixel Filter**: Customize film parameters or reconstruction
        filters via `mitsuba:sensor:film:*` (e.g.,
        `mitsuba:sensor:film:rfilter:type` or
        `mitsuba:sensor:film:pixel_filter:type`).
    *   **Surface Sensors**: Attach a surface sensor (e.g., `irradiancemeter`)
        to a geometry prim by setting `mitsuba:sensor:shape` (string or asset
        path) on the `Camera` prim. This allows measuring light hitting that
        specific surface (e.g., for irradiance mapping). An example is provided
        in `test_assets/shapes`.
*   **Motion Blur**: Transformation motion blur is supported for cameras, non-area lights (point, spot, distant, and dome lights), and geometry (meshes, implicit surfaces, and point instancers, wrapped in Mitsuba instances when their transform is animated) via standard USD time-sampled transforms and camera `shutter:open` / `shutter:close` attributes. The maximum number of transform samples over the shutter interval can be overridden by authoring the `primvars:mitsuba:motion_samples` (int, default `32`) primvar on a prim or any of its ancestors. Area lights, emissive meshes, meshes with an attached surface sensor, and deformation motion blur are not supported; these render with their static transform at frame time.

### Hydra AOVs (Arbitrary Output Variables)
The `color` AOV is RGB (3 channels, `Float32Vec3`) by default. To also render an alpha channel, request a 4-channel buffer, e.g., by setting the `dataType` of the color `RenderVar` to `color4f` (or `float4`). Alpha then holds pixel coverage (`0` where no geometry is visible). Alpha is only rendered when requested, since the extra film channel makes film accumulation noticeably slower.

When requested by the Hydra host, `hdMitsuba` automatically routes and maps standard AOVs to Mitsuba's AOV integrator. It supports the following standard USD AOVs:
*   `depth` $\rightarrow$ `depth:depth` (1 channel)
*   `normal` $\rightarrow$ `sh_normal:sh_normal` (3 channels)
*   `primId` / `instanceId` $\rightarrow$ `shape_index` (1 channel)
*   `elementId` $\rightarrow$ `prim_index` (1 channel)

Additionally, Mitsuba's native AOV names are also supported (e.g., `albedo`, `diffuse_albedo`, `specular_reflectance`, `specular_transmittance`, `roughness`, `position`, `uv`, `geo_normal`, `sh_normal`, `dp_du`, `dp_dv`).

## 2. Render Settings Configuration

You can configure the render delegate by authoring a `RenderSettings` prim inside your USD stage, or by passing settings programmatically.

### Custom Mitsuba Settings
*   **`mitsuba:variant`**: The Mitsuba variant to use for rendering (e.g., `scalar_rgb`, `cuda_ad_rgb`, `llvm_ad_rgb`).
*   **`mitsuba:sample_count`**: The target samples per pixel (SPP) for offline
    renders (alias for `mitsuba:sampler:sample_count`; if both are authored on
    `RenderSettings`, `mitsuba:sampler:sample_count` takes precedence).
*   **`mitsuba:sampler:type`** & **`mitsuba:sampler:<param>`**: Configures the
    Mitsuba sampler plugin (e.g., `independent`, `stratified`, `multijitter`,
    `orthogonal`, `ldsampler`) and its parameters (e.g.,
    `mitsuba:sampler:jitter`, `mitsuba:sampler:seed`). In interactive
    progressive mode (`enableInteractive = true`), structured samplers
    automatically fall back to `independent` so multi-pass accumulation remains
    unbiased.
*   **`mitsuba:seed`**: The random number generator seed offset (int, default `0`) passed to the Mitsuba integrator. Changing `mitsuba:seed` resets progressive accumulation.
*   **`mitsuba:interactive_samples_per_pass`**: Number of samples per pixel rendered in each Hydra pass while `enableInteractive` is `true` (default `1`). Larger values reduce per-pass overhead (and, with kernel freezing enabled, are replayed as a single frozen kernel) at the cost of coarser progressive updates. Changing `mitsuba:interactive_samples_per_pass` or `mitsuba:sample_count` resets progressive accumulation.
*   **`mitsuba:integrator:type`**: The Mitsuba integrator to use (e.g., `path`, `volpath`, `aov`, `direct`).
*   **`mitsuba:integrator:<param>`**: Any flat or nested setting in the
    `mitsuba:integrator:` namespace is forwarded to the Mitsuba integrator
    (e.g., `mitsuba:integrator:max_depth`, `mitsuba:integrator:rr_depth`, or
    nested child integrators). Unset properties use the Mitsuba integrator's
    defaults, and properties not supported by the integrator are ignored with a
    warning. Changing these settings recreates the integrator and resets
    progressive accumulation.
*   **`mitsuba:use_kernel_freezing`**: Enables Dr.Jit's kernel freezing. When enabled, the JIT compilation is frozen after the first frame, drastically reducing JIT tracing overhead for subsequent frames (extremely beneficial for interactive camera navigation in viewports). This is currently disabled by default, as it is still in a somewhat experimental state in its Hydra integration, and also will not work in all Mitsuba 3 scenes.
*   **`mitsuba:denoise`**: Denoises the rendered `color` image (bool, default
    `false`) using NVIDIA OptiX on CUDA variants (`cuda_*`) or Intel
    OpenImageDenoise on CPU/LLVM/Metal variants (`scalar_*`, `llvm_*`,
    `metal_*`), optionally guided by the `albedo` and `sh_normal`/`normal` AOVs
    when requested (normal guidance requires `albedo`). Enabling
    `mitsuba:denoise` forces the `box` pixel filter so pixel noise remains
    uncorrelated.

### Hierarchical Mitsuba Plugin Specs (`mitsuba:<plugin>:*`)

Both `hdMitsuba` and the offline `usd_mitsuba` translator share a uniform USD
namespace convention for configuring native Mitsuba plugins and nested child
plugins:

*   **Supported Namespaces**: `mitsuba:integrator:*` and `mitsuba:sampler:*` on
    `RenderSettings` prims; `mitsuba:sensor:*` on `Camera` prims; and
    `mitsuba:emitter:*` and `mitsuba:shape:*` on supported USD light and
    geometry prims (`Mesh`, `Sphere`, `Cylinder`, `Cube`, and `Plane`).
*   **Hierarchy & Plugin Selection**: Colon-separated attribute names map to
    nested Mitsuba plugin properties. `:type` selects a plugin type, `:id`
    assigns a Mitsuba object ID, and nested `:child:type` / `:child:<param>`
    entries recursively instantiate child plugins (for example,
    `mitsuba:sensor:film:rfilter:type = "gaussian"` and
    `mitsuba:sensor:film:rfilter:stddev = 0.5`).
*   **Value Conversions**: `color3f` attributes convert to RGB spectra, spatial
    `point3f` / `vector3f` / `normal3f` attributes convert to Mitsuba 3-vectors,
    `matrix3d` / `matrix4d` convert to Mitsuba transforms, `asset` paths resolve
    through USD's asset resolver, and inline child specs support
    `:type = "rgb"`, `"spectrum"` (wavelength:weight pairs or uniform float),
    and `"vector"`.

### Standard Hydra Viewport Settings
*   **`enableInteractive`**:
    *   When `true` (default and in interactive viewports), the delegate runs in **progressive mode** (rendering sample-by-sample and clearing the accumulator instantly when the camera or objects move). Rendering each sample progressively has a high JIT tracing cost (unless kernel freezing is enabled), which means this mode should only be used in interactive viewports.
    *   When `false` (in offline batch renderers, e.g., our Python render engine), the delegate renders all samples at once. Note that due to Mitsuba's JIT tracing, this is expected to be *significantly* faster than interactive mode.


## 3. Authoring Example

To configure Mitsuba rendering parameters directly inside a USD file, define a `RenderSettings` prim as follows:

```usd
#usda 1.0

def RenderSettings "/Render/MitsubaSettings"
{
    # Standard USD settings
    rel camera = </World/main_camera>
    int2 resolution = (1920, 1080)

    # Custom Mitsuba settings
    custom string mitsuba:variant = "llvm_ad_rgb"
    custom string mitsuba:sampler:type = "multijitter"
    custom int mitsuba:sampler:sample_count = 256
    custom int mitsuba:seed = 0
    custom int mitsuba:interactive_samples_per_pass = 4
    custom string mitsuba:integrator:type = "path"
    custom int mitsuba:integrator:max_depth = 8

    # Enable Dr.Jit kernel freezing for fast interactive viewport updates
    custom bool mitsuba:use_kernel_freezing = true
}
```

To render this scene using the CLI or `usdrecord`, ensure the `RenderSettings` prim path is set as the active render settings in your layer metadata:
```usd
#usda 1.0
(
    renderSettingsPrimPath = "/Render/MitsubaSettings"
)
```
