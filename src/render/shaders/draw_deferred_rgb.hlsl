#include "shader_utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

// This is an array of all the textures
[[vk::binding(0, 0)]]
Texture2D<float4> rgbInBuffer[];

[[vk::binding(1, 0)]]
Texture2D<float> depthInBuffer[];

[[vk::binding(2, 0)]]
Texture2D<float4> normalInBuffer[];

[[vk::binding(3, 0)]]
Texture2D<int> segmentationInBuffer[];


[[vk::binding(4, 0)]]
RWStructuredBuffer<uint32_t> rgbOutputBuffer;

[[vk::binding(5, 0)]]
RWStructuredBuffer<float> depthOutputBuffer;

[[vk::binding(6, 0)]]
RWStructuredBuffer<uint32_t> normalOutputBuffer;

[[vk::binding(7, 0)]]
RWStructuredBuffer<int> segmentationOutputBuffer;


[[vk::binding(8, 0)]]
SamplerState linearSampler;

// Instances and views
[[vk::binding(0, 1)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 1)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 1)]]
StructuredBuffer<uint32_t> instanceOffsets;

// Lighting
[[vk::binding(0, 2)]]
StructuredBuffer<RenderOptions> renderOptionsBuffer;


#include "lighting.h"

float calculateLinearDepth(float depth_in, uint view_idx)
{
    // Calculate linear depth with reverse-z buffer
    PerspectiveCameraData cam_data = unpackViewData(viewDataBuffer[view_idx]);
    float z_near = cam_data.zNear;
    float z_far = cam_data.zFar;
    float linear_depth = (z_near * z_far) / (z_far + depth_in * (z_near - z_far));
    return linear_depth;
}

uint32_t float3ToUint32(float3 v)
{
    return (uint32_t)(v.x * 255.0f) | ((uint32_t)(v.y * 255.0f) << 8) | ((uint32_t)(v.z * 255.0f) << 16) | (255 << 24);
}

float linearToSRGB(float v)
{
    if (v <= 0.00031308f) {
        return 12.92f * v;
    } else {
        return 1.055f*pow(v,(1.f / 2.4f)) - 0.055f;
    }
}

uint32_t linearToSRGB8(float3 rgb)
{
    float3 srgb = float3(
        linearToSRGB(rgb.x), 
        linearToSRGB(rgb.y), 
        linearToSRGB(rgb.z));

    return float3ToUint32(srgb);
}

uint3 getPixelOffset(uint view_idx)
{
    uint num_views_per_image =
        pushConst.maxImagesXPerTarget * pushConst.maxImagesYPerTarget;

    uint target_idx = view_idx / num_views_per_image;

    uint target_view_idx = view_idx % num_views_per_image;
    uint target_view_idx_x = target_view_idx % pushConst.maxImagesXPerTarget;
    uint target_view_idx_y = target_view_idx / pushConst.maxImagesXPerTarget;

    uint x_pixel_offset = target_view_idx_x * pushConst.viewWidth;
    uint y_pixel_offset = target_view_idx_y * pushConst.viewHeight;

    return uint3(x_pixel_offset, y_pixel_offset, target_idx);
}

// idx.x is the x coordinate of the image
// idx.y is the y coordinate of the image
// idx.z is the global view index
[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint view_idx = idx.z;
    if (idx.x >= pushConst.viewWidth || idx.y >= pushConst.viewHeight) {
        return;
    }

    uint32_t out_pixel_idx =
        view_idx * pushConst.viewWidth * pushConst.viewHeight +
        idx.y * pushConst.viewWidth + idx.x;

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[view_idx]);

    uint3 pixel_offset = getPixelOffset(view_idx);
    uint target_idx = pixel_offset.z;
    float2 view_pixel_offset = float2(pixel_offset.xy);

    float2 sample_px = float2(idx.xy);
    bool sample_valid = true;

    if (view_data.projectionType == MADRONA_PROJECTION_FISHEYE_EQUIDISTANT) {
        float2 uv = (float2(idx.xy) + 0.5f) /
                    float2(pushConst.viewWidth, pushConst.viewHeight);
        float2 ndc = float2(uv.x * 2.0f - 1.0f,
                            1.0f - uv.y * 2.0f);

        float aspect = max(view_data.aspectRatio, 1e-6f);
        float2 aspect_scale = aspect >= 1.0f ?
            float2(1.0f / aspect, 1.0f) :
            float2(1.0f, aspect);

        float2 scaled = float2(ndc.x / aspect_scale.x,
                               ndc.y / aspect_scale.y);
        float norm_r = length(scaled);

        if (norm_r > 1.0f) {
            sample_valid = false;
        } else {
            const float rim_cutoff = 0.999f;
            if (norm_r >= rim_cutoff) {
                sample_valid = false;
            }

            float theta_max = max(view_data.fisheyeThetaMax, 1e-4f);
            float theta = norm_r * theta_max;
            float sin_theta = sin(theta);
            float cos_theta = cos(theta);

            float2 plane_dir = norm_r > 1e-5f ?
                scaled / max(norm_r, 1e-6f) :
                float2(0.0f, 0.0f);

            float3 dir = float3(
                plane_dir.x * sin_theta,
                cos_theta,
                plane_dir.y * sin_theta);

            if (dir.y <= 1e-5f) {
                sample_valid = false;
            } else {
                float persp_x = (dir.x / dir.y) * view_data.xScale;
                float persp_y = (dir.z / dir.y) * view_data.yScale;
                float2 persp_ndc = float2(persp_x, persp_y);

                if (abs(persp_ndc.x) > 1.0f || abs(persp_ndc.y) > 1.0f) {
                    sample_valid = false;
                } else {
                    float view_w = float(pushConst.viewWidth);
                    float view_h = float(pushConst.viewHeight);
                    sample_px = float2(
                        (persp_ndc.x * 0.5f + 0.5f) * view_w,
                        (persp_ndc.y * 0.5f + 0.5f) * view_h - 1.0f);
                }
            }
        }
    }

    if (sample_valid) {
        float2 min_px = float2(0.0f, 0.0f);
        float2 max_px = float2(
            max(float(pushConst.viewWidth) - 1.0f, 0.0f),
            max(float(pushConst.viewHeight) - 1.0f, 0.0f));
        sample_px = clamp(sample_px, min_px, max_px);
    }

    float2 atlas_px = sample_px + view_pixel_offset;
    float2 atlas_min = view_pixel_offset;
    float2 atlas_extent = max(
        float2(float(pushConst.viewWidth),
               float(pushConst.viewHeight)) - float2(1.0f, 1.0f),
        float2(0.0f, 0.0f));
    float2 atlas_max = atlas_min + atlas_extent;
    float2 atlas_clamped = clamp(atlas_px, atlas_min, atlas_max);

    if (renderOptionsBuffer[0].outputs[0]) {
        float3 out_color = float3(0.0f, 0.0f, 0.0f);

        if (sample_valid) {
            uint2 rgb_dims;
            rgbInBuffer[target_idx].GetDimensions(rgb_dims.x, rgb_dims.y);
            float2 rgb_dims_f = float2(rgb_dims);
            float2 half_pixel_uv = float2(
                0.5f / max(rgb_dims_f.x, 1.0f),
                0.5f / max(rgb_dims_f.y, 1.0f));
            float2 atlas_uv = clamp(
                (atlas_clamped + 0.5f) / rgb_dims_f,
                half_pixel_uv,
                float2(1.0f, 1.0f) - half_pixel_uv);
            float4 color = rgbInBuffer[target_idx].SampleLevel(
                linearSampler, atlas_uv, 0);
            out_color = color.rgb;
        }

        rgbOutputBuffer[out_pixel_idx] = linearToSRGB8(out_color);
    }

    if (renderOptionsBuffer[0].outputs[1]) {
        float linear_depth = view_data.zFar;

        if (sample_valid) {
            uint2 depth_dims;
            depthInBuffer[target_idx].GetDimensions(depth_dims.x,
                                                    depth_dims.y);
            float2 depth_dims_f = float2(depth_dims);
            float2 half_pixel_uv = float2(
                0.5f / max(depth_dims_f.x, 1.0f),
                0.5f / max(depth_dims_f.y, 1.0f));
            float2 atlas_uv = clamp(
                (atlas_clamped + 0.5f) / depth_dims_f,
                half_pixel_uv,
                float2(1.0f, 1.0f) - half_pixel_uv);
            float depth_in = depthInBuffer[target_idx].SampleLevel(
                linearSampler, atlas_uv, 0).x;
            linear_depth = calculateLinearDepth(depth_in, view_idx);
        }

        depthOutputBuffer[out_pixel_idx] = linear_depth;
    }

    if (renderOptionsBuffer[0].outputs[2]) {
        float3 normal_sample = float3(0.0f, 0.0f, 0.0f);

        if (sample_valid) {
            uint2 normal_dims;
            normalInBuffer[target_idx].GetDimensions(normal_dims.x,
                                                     normal_dims.y);
            float2 normal_dims_f = float2(normal_dims);
            float2 half_pixel_uv = float2(
                0.5f / max(normal_dims_f.x, 1.0f),
                0.5f / max(normal_dims_f.y, 1.0f));
            float2 atlas_uv = clamp(
                (atlas_clamped + 0.5f) / normal_dims_f,
                half_pixel_uv,
                float2(1.0f, 1.0f) - half_pixel_uv);
            float4 normal_tex = normalInBuffer[target_idx].SampleLevel(
                linearSampler, atlas_uv, 0);
            normal_sample = normal_tex.xyz;
        }

        normalOutputBuffer[out_pixel_idx] = float3ToUint32(normal_sample);
    }
    
    if (renderOptionsBuffer[0].outputs[3]) {
        int segmentation = -1;

        if (sample_valid) {
            int2 atlas_px_i = int2(atlas_clamped + 0.5f);
            segmentation = segmentationInBuffer[target_idx].Load(
                int3(atlas_px_i, 0));
        }

        segmentationOutputBuffer[out_pixel_idx] = segmentation;
    }
}
