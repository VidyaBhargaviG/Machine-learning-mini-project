from django.shortcuts import render
# Create your views here.
def my_view(request):
    d={"name":"Sachin","age":23}
    return render(request, 'myapp/index.html',d)

