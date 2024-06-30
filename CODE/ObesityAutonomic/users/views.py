# Create your views here.
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    path = settings.MEDIA_ROOT+"//"+'heart_obecity.csv'
    import pandas as pd
    df = pd.read_csv(path)
    df =df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


def user_cart_analysis(request):
    from .utility.ProcessCart import start_process_cart
    rslt_dict = start_process_cart()
    return render(request, "users/cartresults.html", rslt_dict)


def user_gbdt_analysis(request):
    from.utility.ProcessCart import start_process_gbdt
    rslt_dict = start_process_gbdt()
    return render(request, "users/gbdtresults.html", rslt_dict)



def user_predictions(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        anaemia = int(request.POST.get('anaemia'))
        creatinine_phosphokinase = int(request.POST.get('creatinine_phosphokinase'))
        diabetes = int(request.POST.get('diabetes'))
        ejection_fraction = float(request.POST.get('ejection_fraction'))
        high_blood_pressure = float(request.POST.get('high_blood_pressure'))
        platelets = int(request.POST.get('platelets'))
        serum_creatinine = float(request.POST.get('serum_creatinine'))
        serum_sodium = int(request.POST.get('serum_sodium'))
        sex = int(request.POST.get('sex'))
        smoking = int(request.POST.get('smoking'))
        time = int(request.POST.get('time'))

        test_data = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]
        from .utility.predections import test_user_data
        test_pred = test_user_data(test_data)
        if test_pred[0] == 0:
            rslt = False
        else:
            rslt = True
        return render(request, "users/testform.html", {"test_data": test_data, "result": rslt})


    else:
        return render(request, "users/testform.html", {})
