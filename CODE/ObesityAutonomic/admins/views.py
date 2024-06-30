from django.shortcuts import render, HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})


def AdminCartResults(request):
    from users.utility.ProcessCart import start_process_cart
    rslt_dict = start_process_cart()
    return render(request, "admins/admincartresults.html", rslt_dict)


def AdminGBDTResults(request):
    from users.utility.ProcessCart import start_process_gbdt
    rslt_dict = start_process_gbdt()
    return render(request, "admins/admingbdtresults.html", rslt_dict)
