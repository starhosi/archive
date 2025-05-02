from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
app_name = 'opencv_webapp'

urlpatterns=[
    path('',views.first_view,name='first_view'),
    path('upload_image/',views.simple_upload,name='simple_upload'),
    path('detect_face/',views.detect_face,name='detect_face'),
    # path('<int:question_id>/',views.detail,name='detail'),
    # path('<int:question_id>/result/',views.result,name='result'),
    # path('<int:question_id>/vote/',views.vote,name='vote'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
