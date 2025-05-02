from django.shortcuts import render
from .crawling import crawling_result
from . import crawling

# Create your views here.

def index(request):
    return render(request, 'semi_webapp/index.html',{})

def result(request):
    keyword_submitted = str(request.POST['input_location'] + ' ' + request.POST['input_category'])
    craw = crawling_result((request.POST['input_location'], request.POST['input_category']))

    context={'keyword': keyword_submitted, 'crawl_result': craw}
    # context={'keyword': keyword_submitted, 'store': craw[0], 'score':craw[1], 'review': craw[2]}
    return render(request, 'semi_webapp/result.html', context)
