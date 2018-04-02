
#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
//#include <iomanip.h>
#include "cudaNormalize.h"
#include "cudaFont.h"
#include "imageNet.h"


#define DEFAULT_CAMERA 1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
#define DEFAULT_CAMERAA 2		
		
		
bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	printf("imagenet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	gstCamera* camera2 = gstCamera::Create(DEFAULT_CAMERAA);
	if( !camera )
	{
		printf("\nimagenet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	if( !camera2 )
	{
		printf("\nimagenet-camera2:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\nimagenet-camera2:  successfully initialized video device\n");
	printf("    width2:  %u\n", camera2->GetWidth());
	printf("   height2:  %u\n", camera2->GetHeight());
	printf("    depth2:  %u (bpp)\n\n", camera2->GetPixelDepth());
	/*
	 * create imageNet
	 */
	imageNet* net = imageNet::Create(argc, argv);
	imageNet* net2 = imageNet::Create(argc, argv);
	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\nimagenet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("imagenet-camera:  failed to create openGL texture\n");
	}
	
	glTexture* texture2 = NULL;
	
	if( !display ) {
		printf("\nimagenet-camera2:  failed to create openGL display2\n");
	}
	else
	{
		texture2 = glTexture::Create(camera2->GetWidth(), camera2->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture2 )
			printf("imagenet-camera2:  failed to create openGL2 texture2\n");
	}
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	cudaFont* font2 = cudaFont::Create();	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nimagenet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  camera open for streaming\n");
	
	if( !camera2->Open() )
	{
		printf("\nimagenet-camera2:  failed to open2 camera for streaming\n");
		return 0;
	}

	printf("\nimagenet-camera2:  camera2 open for streaming\n");
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	float confidence2 = 0.0f;

	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\nimagenet-camera:  failed to capture frame\n");
		//else
		//	printf("imagenet-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
		
		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
			printf("imagenet-camera:  failed to convert from NV12 to RGBA\n");

		// classify image
		const int img_class = net->Classify((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);
	
		if( img_class >= 0 )
		{
			printf("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));	

			if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
	
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 10, make_float4(255.0f, 0.0f, 05.0f, 255.0f));
			}
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, net->GetNetworkName(), net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "TensorRT build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(200,50);		
			}

			//display->EndRender();
		}


		void* imgCPU2  = NULL;
		void* imgCUDA2 = NULL;

// get the latest frame
		if( !camera2->Capture(&imgCPU2, &imgCUDA2, 1000) )          //1000 is time 
			printf("\nimagenet-camera2:  failed to capture frame\n");
		//else
		//	printf("imagenet-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
		// convert from YUV to RGBA=======duplicate

		void* imgRGBA2 = NULL;
		
		if( !camera2->ConvertRGBA(imgCUDA2, &imgRGBA2) )
			printf("imagenet-camera2:  failed to convert from NV12 to RGBA\n");

		// classify image
		const int img_class2 = net2->Classify((float*)imgRGBA2, camera2->GetWidth(), camera2->GetHeight(), &confidence2);
	
		if( img_class2 >= 0 )
		{
			printf("imagenet-camera2:  %2.5f%% class #%i (%s)\n", confidence2 * 100.0f, img_class2, net2->GetClassDesc(img_class2));	//print to terminal 

			if( font2 != NULL )
			{
				char str2[256];
				sprintf(str2, "%05.2f%% %s", confidence2 * 100.0f, net2->GetClassDesc(img_class2)); //print to display
	
				font2->RenderOverlay((float4*)imgRGBA2, (float4*)imgRGBA2, camera2->GetWidth(), camera2->GetHeight(),
								    str2, 10, 10, make_float4(255.0f, 0.0f, 26.0f, 255.0f));
			}
			
		//	if( display != NULL )
		//	{
				//char str2[256];
				//sprintf(str2, "TensorRT build2 %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, net2->GetNetworkName(), net2->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "TensorRT build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				//display->SetTitle(str2);	
			//}	
		}	

// update display=======duplicate
		if( display != NULL )
		{
			//display->UserEvents();
			//display->BeginRender();

			if( texture2 != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA2, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA2, make_float2(0.0f, 1.0f), 
		 						   camera2->GetWidth(), camera2->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map2 = texture2->MapCUDA();

				if( tex_map2 != NULL )
				{
					cudaMemcpy(tex_map2, imgRGBA2, texture2->GetSize(), cudaMemcpyDeviceToDevice);
					texture2->Unmap();
				}

				// draw the texture
				texture2->Render(200,500);		
			}

			display->EndRender();
		}
	}
	
	printf("\nimagenet-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}

	
	printf("imagenet-camera:  video device has been un-initialized.\n");
	printf("imagenet-camera:  this concludes the test of the video device.\n");

	/*
	 * shutdown the camera device
	 */
	if( camera2 != NULL )
	{
		delete camera2;
		camera2 = NULL;
	}

	//if( display2 != NULL )

	//{
	//	delete display2;
	//	display2 = NULL;
	//}
	
	printf("imagenet-camera2:  video device has been un-initialized.\n");
	printf("imagenet-camera2:  this concludes the test of the video device.\n");
	return 0;
}
