# deployment/views.py

from django.shortcuts import render
from django.conf import settings
from .model import predict
import os
import traceback


def upload_video(request):
    context = {}
    if request.method == 'POST':
        uploaded_video = request.FILES.get('video')
        if uploaded_video:
            video_path = os.path.join(settings.MEDIA_ROOT, uploaded_video.name)
            print(f"Uploading video to: {video_path}")

            try:
                with open(video_path, 'wb+') as destination:
                    for chunk in uploaded_video.chunks():
                        destination.write(chunk)
                print("Video uploaded successfully.")
            except Exception as e:
                print(f"Error saving video: {e}")
                print(traceback.format_exc())
                context['prediction'] = "Error saving video."
                return render(request, 'deployment/upload.html', context)

            if os.path.exists(video_path):
                print(f"Video saved successfully at: {video_path}")

                try:
                    prediction = predict(video_path)
                    print(f"Prediction: {prediction}")
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    print(traceback.format_exc())
                    prediction = "Error during prediction."

                try:
                    os.remove(video_path)
                    print("Video file removed after prediction.")
                except Exception as e:
                    print(f"Error removing video file: {e}")
                    print(traceback.format_exc())

                context['prediction'] = prediction
            else:
                print(f"Failed to save video at: {video_path}")
                context['prediction'] = "Failed to upload video."
        else:
            print("No video file was uploaded.")
            context['prediction'] = "No video file was uploaded."
    return render(request, 'deployment/upload.html', context)
