from django import forms
from .models import ImageUploadModel

# form / models 차이점 이해하기
# form : 이미지 파일을 업로드 받아 프로젝트의 mdeia 디렉토리에 저장하는 역할까지만 수행
# model : 이미지를 업로드받아 DB 저장뒤, Opencv처리까지 적용.
class SimpleUploadForm(forms.Form):
    title = forms.CharField(max_length=50)
    image = forms.ImageField()

class ImageUploadForm(forms.ModelForm):
    # DB연결
    class Meta:
        model = ImageUploadModel
        fields = ('description','document',)
