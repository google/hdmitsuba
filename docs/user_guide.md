# hdMitsuba User Guide: Supported Features & Render Settings

This document provides an overview of the USD features supported by `hdMitsuba`, custom render settings, and how to author USD scenes for rendering with Mitsuba 3. Most features are covered by unit tests, so the `test_assets` folder is also a good reference.

## 1. Supported USD Features

`hdMitsuba` translates many standard USD concepts into Mitsuba equivalents. We are continuously expanding the supported feature set, currently we support:

### Geometry & Animation
*   **Meshes**: Flat meshes, subdivision surfaces (Catmull-Clark), and deforming geometry (such as skeletal animation). Subdivision surfaces are evaluated at a fixed subdivision level before rendering (no on-the-fly or adaptive tessellation). The subdivision level can be overridden per mesh by authoring the custom `mitsuba:subdivision_level` (int) attribute.
*   **Curves**: Linear and cubic curves (translated to Mitsuba curve shapes). USD supports additional curve types (e.g., Bezier) which Mitsuba itself currently does not support.
*   **Instancing**: Full support for USD point instancers and instanced geometry references. Note that Mitsuba's instances cannot be light sources currently.
*   **Displacement Mapping**: Fully supports displacement mapping. The delegate evaluates connected displacement textures from both standard USD displacement terminals and `mitsuba:displacement` terminals.

### Shading & Materials
*   **UsdPreviewSurface**: Various USD preview surface parameters are mapped to Mitsuba's `principled` BSDF:
    *   `diffuseColor` (Base color)
    *   `roughness` (Specular roughness)
    *   `metallic` (Metalness)
    *   `specular` (Specular reflectance)
    *   `clearcoat` & `clearcoatRoughness`
    *   `opacity` (Transmission)
    *   `ior` (Index of refraction)
    *   `normal` (Normal mapping)
*   **Native Mitsuba BSDFs & Textures**: We support embedding native Mitsuba BSDFs and textures as custom USD shader prims. This allows accessing all of Mitsuba's built-in BSDFs. Examples for this are provided in `test_assets/materials`. Note that USD natively allows specifying both a generic preview surface and a renderer-specific material. hdMitsuba will automatically use the shader that is connected to `outputs:mitsuba:surface`, if available. 
*   **Fallback Shading (displayColor)**: If a mesh has no bound material, the delegate automatically maps the USD `displayColor` attribute to a fallback `diffuse` BSDF with its `reflectance` bound to the color.

### Lights & Camera
*   **Light Types**: Sphere lights, point lights, spot lights, rectangle lights, distant lights, and dome lights (for HDR environment map backgrounds). Currently, IES profiles are not yet supported.
*   **Cameras & Custom Sensors**:
    *   Supports standard perspective cameras with horizontal/vertical aperture offsets.
    *   **Custom Sensor Type**: Customize the Mitsuba sensor backend by setting `mitsuba:sensor:type` (string, e.g., `perspective`, `irradiancemeter`, `radmeter`) on the Camera prim.
    *   **Pixel Filter**: Customize the film's pixel filter by setting `mitsuba:sensor:film:pixel_filter:type` (string) on the Camera prim.
    *   **Surface Sensors**: Attach a sensor to a Mesh geometry prim by setting the `mitsuba:sensor` (string/path) attribute on the Mesh. This allows measuring light hitting that specific surface (e.g., for irradiance mapping). An example is provided in `test_assets/shapes`.

### Hydra AOVs (Arbitrary Output Variables)
When requested by the Hydra host, `hdMitsuba` automatically routes and maps standard AOVs to Mitsuba's AOV integrator. It supports the following standard USD AOVs:
*   `depth` $\rightarrow$ `depth:depth` (1 channel)
*   `normal` $\rightarrow$ `sh_normal:sh_normal` (3 channels)
*   `primId` / `instanceId` $\rightarrow$ `shape_index` (1 channel)
*   `elementId` $\rightarrow$ `prim_index` (1 channel)

Additionally, Mitsuba's native AOV names are also supported (e.g., `duv_dx`).

## 2. Render Settings Configuration

You can configure the render delegate by authoring a `RenderSettings` prim inside your USD stage, or by passing settings programmatically.

### Custom Mitsuba Settings
*   **`mitsuba:variant`**: The Mitsuba variant to use for rendering (e.g., `scalar_rgb`, `cuda_ad_rgb`, `llvm_ad_rgb`).
*   **`mitsuba:sample_count`** : The target samples per pixel (SPP) for high-quality offline renders.
*   **`mitsuba:integrator:type`**: The Mitsuba integrator to use (e.g., `path`, `aov`, `direct`).
*   **`mitsuba:use_kernel_freezing`**: Enables Dr.Jit's kernel freezing. When enabled, the JIT compilation is frozen after the first frame, drastically reducing JIT tracing overhead for subsequent frames (extremely beneficial for interactive camera navigation in viewports). This is currently disabled by default, as it is still in a somewhat experimental state in its Hydra integration, and also will not work in all Mitsuba 3 scenes (e.g., with instancing).

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
    custom int mitsuba:sample_count = 256
    custom string mitsuba:integrator:type = "path"

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
