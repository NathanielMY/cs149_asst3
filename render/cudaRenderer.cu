#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include <iostream>
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
	return n;
}

//copied from scan.cu
__global__ void
getpositions(int *mask, int *scanned_mask, int N)
{ 
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (t_idx >= N-1) { return; }
	
	mask[t_idx] *= scanned_mask[t_idx + 1];
}
//copied from scan.cu
__global__ void
writeindices(int *positions, int *result, int N)
{
    
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (t_idx >= N-1) { return; }

	if (positions[t_idx] != 0) {
		result[positions[t_idx]-1] = t_idx;
	}
}

//adapted for the tensor
__global__ void
tensor_getpositions(int *mask_tensor, int *scanned_tensor, int rounded_num_circles_in_tile, int num_pixels)
{ 
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (t_idx >= rounded_num_circles_in_tile * num_pixels - 1) { return; }

    int pixel_idx = t_idx / rounded_num_circles_in_tile;
    int circle_idx = t_idx % rounded_num_circles_in_tile;

    int pixel_offset = rounded_num_circles_in_tile * pixel_idx;

	
	mask_tensor[pixel_offset + circle_idx] *= scanned_tensor[pixel_offset + circle_idx + 1];
}

//adapted for the tensor
__global__ void
tensor_writeindices(int *mask_tensor, int *scanned_tensor, int rounded_num_circles_in_tile, int num_pixels)
{
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (t_idx >= rounded_num_circles_in_tile * num_pixels - 1) { return; }

    int pixel_idx = t_idx / rounded_num_circles_in_tile;
    int circle_idx = t_idx % rounded_num_circles_in_tile;

    int pixel_offset = rounded_num_circles_in_tile * pixel_idx;
	//int *pixel_scanned_tensor = scanned_tensor + rounded_num_circles_in_tile * pixel_idx;

	if (mask_tensor[pixel_offset + circle_idx] != 0) {
		scanned_tensor[pixel_offset + mask_tensor[pixel_offset + circle_idx]-1] = circle_idx;
	}
}

__global__ void
scan_upsweep(int* buffer, int N, int sourceStepSize, int destinationStepSize)
{
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	// if (t_idx * destinationStepSize + destinationStepSize - 1 >= N) { return; }
	if (t_idx >= N / destinationStepSize) { return; }
	
	unsigned long write_idx = t_idx * destinationStepSize;
	buffer[write_idx + destinationStepSize - 1] += buffer[write_idx + sourceStepSize - 1];
}

__global__ void
scan_downsweep(int* buffer, int N, int sourceStepSize, int destinationStepSize)
{
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	// if (t_idx * destinationStepSize + destinationStepSize - 1 >= N) { return; }
	if (t_idx >= N / destinationStepSize) { return; }

	unsigned long write_idx = t_idx * destinationStepSize;
	if (t_idx == 0 && destinationStepSize == N) { buffer[N-1] = 0; }

	int tmp = buffer[write_idx + sourceStepSize - 1];
	buffer[write_idx + sourceStepSize - 1] = buffer[write_idx + destinationStepSize - 1];
	buffer[write_idx + destinationStepSize - 1] += tmp;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result, int THREADS_PER_BLOCK)
{
	// cudaMemcpy(result, input, N*sizeof(int), cudaMemcpyDeviceToDevice);
	// N = nextPow2(N);

	// // upsweep
	// int destinationStepSize = 1;
	// for (int sourceStepSize= 1; sourceStepSize < N; sourceStepSize = destinationStepSize) {
	// 	destinationStepSize *= 2;
	
	// 	int numThreadsNeeded = N / destinationStepSize;
	// 	int numBlocks = (numThreadsNeeded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	// 	scan_upsweep<<<numBlocks, THREADS_PER_BLOCK>>>(
	// 		result, N, sourceStepSize, destinationStepSize);
	// 	cudaDeviceSynchronize();
	// }

	// // downsweep
	// destinationStepSize = N;
	// for (int sourceStepSize = N/2; sourceStepSize >= 1; sourceStepSize /= 2) {
		
	// 	int numThreadsNeeded = N / destinationStepSize;
	// 	int numBlocks = (numThreadsNeeded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;	
	// 	scan_downsweep<<<numBlocks, THREADS_PER_BLOCK>>>(
	// 		result, N, sourceStepSize, destinationStepSize);
	// 	cudaDeviceSynchronize();

	// 	destinationStepSize = sourceStepSize;
	// }

    cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);

    // Wrap raw pointers with Thrust device pointers
    thrust::device_ptr<int> input_ptr(result);
    thrust::device_ptr<int> result_ptr(result);

    // Perform exclusive scan using Thrust (in-place on `result`)
    thrust::exclusive_scan(input_ptr, input_ptr + N, result_ptr);
    cudaDeviceSynchronize();

}



//updated to support using 1D exclusive scan for the 3D tensor
//adapted for the tensor
__global__ void
adapted_tensor_getpositions(int *mask_tensor, int *scanned_tensor, int rounded_num_circles_in_tile, int num_pixels)
{ 
	unsigned long t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (t_idx >= rounded_num_circles_in_tile * num_pixels - 1) { return; }

    int pixel_idx = t_idx / rounded_num_circles_in_tile;
    int circle_idx = t_idx % rounded_num_circles_in_tile;

    int pixel_offset = rounded_num_circles_in_tile * pixel_idx;

	
	mask_tensor[pixel_offset + circle_idx] *= (scanned_tensor[pixel_offset + circle_idx + 1] - scanned_tensor[pixel_offset]);
}



__global__ void circlesTileMask(
	int num_circles, 
	int rounded_num_circles,
	int *device_output_circles_list,
	int tileWidth, int tileHeight, int imageWidth, int imageHeight)
{
	unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;
    int num_tiles = (imageWidth / tileWidth) * (imageHeight / tileHeight);

	if (idx >= num_circles * num_tiles) { return; }

    int tile_idx = idx / num_circles;
    int circle_idx = idx % num_circles;

    int tile_x_idx = tile_idx % (imageWidth / tileWidth);
    int tile_y_idx = tile_idx / (imageWidth / tileWidth);

	int index3 = 3 * circle_idx;

    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[circle_idx];

	float circle_bottom_left_x = imageWidth * (p.x - rad);
	float circle_bottom_left_y = imageHeight * (p.y - rad);
	float circle_top_right_x = imageWidth * (p.x + rad);
	float circle_top_right_y = imageHeight * (p.y + rad);

    int bottomLeftX = tile_x_idx * tileWidth;
    int bottomLeftY = tile_y_idx * tileHeight;
    int topRightX = bottomLeftX + tileWidth;
    int topRightY = bottomLeftY + tileHeight;

	bool x_overlaps = (bottomLeftX < circle_top_right_x + 1) && (topRightX + 1 > circle_bottom_left_x);
	bool y_overlaps = (bottomLeftY < circle_top_right_y + 1) && (circle_bottom_left_y < topRightY + 1);
    //because we have rounded_num_circles per tile
	device_output_circles_list[tile_idx * rounded_num_circles + circle_idx] = (x_overlaps && y_overlaps) ? 1 : 0; //device_output_circles_list has a list of circles per each pixel
}

__global__ void adapted_copy_count(
		int *count_circles_on_pixel, int* device_scanned_tensor, int num_tiles, 
		int rounded_num_circles) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_tiles) {
		return;
	}
	count_circles_on_pixel[idx] = device_scanned_tensor[idx * rounded_num_circles + rounded_num_circles - 1] - device_scanned_tensor[idx * rounded_num_circles]; //access last value to get total count
}



//TO DO - replace this code so that
void getCirclesInTiles(
	int num_input_circles,
	int **output_circle_list_ptr, int **num_circles_in_tile_list_ptr,
	int tileWidth, int tileHeight, int imageWidth, int imageHeight)
{
	//create an array of circles that overlap each tile. spawn a thread to check for each circle and tile combo
    int num_tiles = (imageWidth / tileWidth) * (imageHeight / tileHeight);

    int *device_output_circle_list;
    int rounded_num_input_circles = nextPow2(num_input_circles + 1);
	cudaMalloc((void **)&device_output_circle_list, sizeof(int)*rounded_num_input_circles * num_tiles);


    int *scan_output_circle_list;
    cudaMalloc((void **)&scan_output_circle_list, sizeof(int) * rounded_num_input_circles * num_tiles);

    int *num_circles_in_tile_list;
    cudaMalloc((void **)&num_circles_in_tile_list, sizeof(int) * num_tiles);

	int threads_per_block = 256;
	int num_blocks = (num_input_circles * num_tiles + threads_per_block - 1) / threads_per_block;
	circlesTileMask<<<num_blocks, threads_per_block>>>(
        num_input_circles, rounded_num_input_circles, device_output_circle_list,
		tileWidth, tileHeight, imageWidth, imageHeight
	);

    cudaDeviceSynchronize();

    // //FLAG - count number of nonzero elements after circlesTileMask
    // int *hostArray = new int[rounded_num_input_circles * num_tiles];
    // cudaMemcpy(hostArray, device_output_circle_list, rounded_num_input_circles * num_tiles * sizeof(int), cudaMemcpyDeviceToHost);

    // int nonZeroCount = 0;
    // for (int i = 0; i < rounded_num_input_circles * num_tiles; i++) {
    //     if (hostArray[i] != 0) {
    //         nonZeroCount++;
    //     }
    // }

    // delete[] hostArray;

    // std::cout << "Number of non-zero elements in the 1's 0's mask tensor: " << nonZeroCount << std::endl;

    exclusive_scan(device_output_circle_list, rounded_num_input_circles * num_tiles, scan_output_circle_list, 256);
	cudaDeviceSynchronize();

    //FLAG TO DO - use the adapted copy code and adapted get positions code

    int num_blocks_for_tiles = (num_tiles + threads_per_block - 1) / threads_per_block;
    adapted_copy_count<<<num_blocks_for_tiles, threads_per_block>>>(
		num_circles_in_tile_list, 
		scan_output_circle_list, 
		num_tiles, 
		rounded_num_input_circles);
	

    num_blocks = (rounded_num_input_circles * num_tiles + threads_per_block - 1) / threads_per_block;
	adapted_tensor_getpositions<<<num_blocks, threads_per_block>>>(
		device_output_circle_list, scan_output_circle_list, rounded_num_input_circles, num_tiles);
	cudaDeviceSynchronize();

	tensor_writeindices<<<num_blocks, threads_per_block>>>(
		device_output_circle_list, scan_output_circle_list, rounded_num_input_circles, num_tiles);
	cudaDeviceSynchronize();

	*output_circle_list_ptr = scan_output_circle_list;
    *num_circles_in_tile_list_ptr = num_circles_in_tile_list;

    cudaFree(device_output_circle_list);
}


// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    
    // simple: each circle has an assigned color
    int index3 = 3 * circleIndex;
    rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    alpha = .5f;
    

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}


__global__ void
shade_per_pixel(int rounded_num_circles, int *circles_on_tile, 
                int *num_circles_in_tile_list, int tileWidth, int tileHeight, int imageWidth, int imageHeight) {

    //FLAG TODO - fetch tile index then do

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = imageWidth * imageHeight;

    if (idx >= num_pixels) return;

    // Calculate pixel coordinates
    int pixelX =(idx % (imageWidth));
    int pixelY = (idx / (imageWidth));

    // Normalized center of the pixel
    float invWidth = 1.f / cuConstRendererParams.imageWidth;
    float invHeight = 1.f / cuConstRendererParams.imageHeight;
    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f)
    );

    //compute the tile location its in
    int x_tile_pos = pixelX / tileWidth;
    int y_tile_pos = pixelY / tileHeight;
    int tile_idx = x_tile_pos + y_tile_pos * (imageWidth / tileWidth);

    // Pointer to the pixel in the image
    float4* imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * cuConstRendererParams.imageWidth + pixelX)]);
    float4 localAccumulator = *imagePtr;  // Start with the current color in the image

    int num_circles = num_circles_in_tile_list[tile_idx];

    // Iterate over circles contributing to this pixel
    for (int x = 0; x < num_circles; ++x) {
        int global_circle_index = circles_on_tile[tile_idx * rounded_num_circles + x];

        // Get circle position
        float3 p = *(float3*)(&cuConstRendererParams.position[global_circle_index * 3]);

        // Call shading function to accumulate circle contribution
        shadePixel(global_circle_index, pixelCenterNorm, p, &localAccumulator);
    }

    // Update pixel color in image
    *imagePtr = localAccumulator;
}



// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
snowShadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}


__global__ void
snow_shade_per_pixel(int rounded_num_circles, int *circles_on_tile, 
                int *num_circles_in_tile_list, int tileWidth, int tileHeight, int imageWidth, int imageHeight) {

    //FLAG TODO - fetch tile index then do

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = imageWidth * imageHeight;

    if (idx >= num_pixels) return;

    // Calculate pixel coordinates
    int pixelX =(idx % (imageWidth));
    int pixelY = (idx / (imageWidth));

    // Normalized center of the pixel
    float invWidth = 1.f / cuConstRendererParams.imageWidth;
    float invHeight = 1.f / cuConstRendererParams.imageHeight;
    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f)
    );

    //compute the tile location its in
    int x_tile_pos = pixelX / tileWidth;
    int y_tile_pos = pixelY / tileHeight;
    int tile_idx = x_tile_pos + y_tile_pos * (imageWidth / tileWidth);

    // Pointer to the pixel in the image
    float4* imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * cuConstRendererParams.imageWidth + pixelX)]);
    float4 localAccumulator = *imagePtr;  // Start with the current color in the image

    int num_circles = num_circles_in_tile_list[tile_idx];

    // Iterate over circles contributing to this pixel
    for (int x = 0; x < num_circles; ++x) {
        int global_circle_index = circles_on_tile[tile_idx * rounded_num_circles + x];

        // Get circle position
        float3 p = *(float3*)(&cuConstRendererParams.position[global_circle_index * 3]);

        // Call shading function to accumulate circle contribution
        snowShadePixel(global_circle_index, pixelCenterNorm, p, &localAccumulator);
    }

    // Update pixel color in image
    *imagePtr = localAccumulator;
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}


//then write a cuda function that for a given pixel and its list of circles, render the pixel


////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

__global__ void initialize_with_index(int *arr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        arr[idx] = idx;
    }
}

void
CudaRenderer::render() {

	struct GlobalConstants params;
	cudaMemcpy(&params, &cuConstRendererParams, sizeof(struct GlobalConstants), 
		cudaMemcpyDeviceToHost);

    bool snow = (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME);

	int num_circles = numCircles;

	std::cout << "number of circles is " << num_circles;
	short image_width = image->width;
	short image_height = image->height;


	//int tile_width = ((int)(image_width / (sqrt(sqrt(num_circles)) * 32))) * 32;
	//int tile_height = ((int)(image_height / (sqrt(sqrt(num_circles)) * 32))) * 32;
	int num_tiles = 16;
	if (num_circles < 100) {
		num_tiles = 1;
	} else if ((1000 < num_circles) && (num_circles < 5000)) {
		num_tiles = 8;
	}


	int tile_width = image_width / num_tiles;
	int tile_height = image_height / num_tiles;
			

    int *device_tile_circles_list;
    int *num_circles_in_tile_list;

    // Time getCirclesInTile
    auto start = std::chrono::high_resolution_clock::now();

    if (num_tiles == 1) {
        cudaMalloc(&device_tile_circles_list, sizeof(int) * num_circles);
        cudaMalloc(&num_circles_in_tile_list, sizeof(int));

        //set device_tile_circles_list so that device_tile_circles_list[idx] = idx
        int threads_per_block = 256;
        int blocks = (num_circles + threads_per_block - 1) / threads_per_block;
        initialize_with_index<<<blocks, threads_per_block>>>(device_tile_circles_list, num_circles);
        cudaDeviceSynchronize();

        cudaMemcpy(num_circles_in_tile_list, &num_circles, sizeof(int), cudaMemcpyHostToDevice);
    } else {
        getCirclesInTiles(num_circles, &device_tile_circles_list, &num_circles_in_tile_list, 
            tile_width, tile_height, image_width, image_height);
    }

    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "getCirclesInTile: " << duration.count() << " ms\n";

    int rounded_num_circles = nextPow2(num_circles + 1);

    // //FLAG - debugging code
    // int numElements = rounded_num_circles * (image_width / tile_width) * (image_height / tile_height);
    // int *hostArray = new int[numElements];

    // // Copy the array from device to host
    // cudaMemcpy(hostArray, device_tile_circles_list, numElements * sizeof(int), cudaMemcpyDeviceToHost);
    

    // int nonZeroCount = 0;
    // for (int i = 0; i < numElements; i++) {
    //     if (hostArray[i] != 0) {
    //         nonZeroCount++;
    //     }
    // }
    // std::cout << "Number of non-zero elements in the scanned tensor: " << nonZeroCount << std::endl;

    // // Free the host array
    // delete[] hostArray;

    int threads_per_block = 256;
    int num_blocks = (image_width * image_height + threads_per_block - 1) / threads_per_block;
    // Time shade_per_pixel kernel launch (if synchronous)
    start = std::chrono::high_resolution_clock::now();

    if (snow) {
        snow_shade_per_pixel<<<num_blocks, threads_per_block>>>(
            rounded_num_circles, device_tile_circles_list,
            num_circles_in_tile_list, tile_width, tile_height, image_width, image_height
        );
    } else {
        shade_per_pixel<<<num_blocks, threads_per_block>>>(
            rounded_num_circles, device_tile_circles_list,
            num_circles_in_tile_list, tile_width, tile_height, image_width, image_height
        );
    }

    

    cudaDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "shade_per_pixel kernel: " << duration.count() << " ms\n";

    cudaFree(device_tile_circles_list);
    cudaFree(num_circles_in_tile_list);


}
