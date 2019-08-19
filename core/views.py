from django.db import transaction
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, DetailView, DeleteView, UpdateView
from .models import Attendance
from users.forms import UserForm, ProfileForm
from django.contrib.auth.models import User
import cv2
import os
import numpy as np
import time
from django.urls import reverse
from django.contrib import messages
from users.models import Profile
from django.urls import reverse_lazy
# from core.model import create_input_image_embeddings, recognize_faces_in_cam, attendance
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create your views here.
class MainDashboard(TemplateView):
    template_name = "core/main_dashboard.html"
    def dispatch(self, request, *args, **kwargs):
        if not self.request.user.is_superuser:
            return HttpResponseRedirect(
                reverse('users:user_dashboard'))
        return super(MainDashboard, self).dispatch(request, *args, **kwargs)

class ManageAttendance(TemplateView):
    template_name = "core/manage_attendance.html"

class BOD(TemplateView):
    template_name = 'core/bod.html'
    model = Profile
    context_object_name = 'profile'

class About(TemplateView):
    template_name = 'core/about.html'

class Contact(TemplateView):
    template_name = 'core/contact.html'

class DeleteEmployee(DeleteView):
    template_name = "core/delete.html"
    model = User
    context_object_name = 'form'

    def get_success_url(self):
        success_url = reverse_lazy('core:employee_record')
        return success_url

class UpdateEmployee(UpdateView):
    model = Profile
    form_class = ProfileForm
    template_name = "core/update.html"

    def get_success_url(self):
        success_url = reverse_lazy('core:employee_record')
        return success_url

class EmployeeRecord(ListView):
    template_name = "core/employee_record.html"
    model = User
    context_object_name = 'employee'

    def get_queryset(self):
        return User.objects.filter(is_superuser=False)

class UserProfile(DetailView):
    template_name = 'core/profile.html'
    model = Profile
    context_object_name = 'profile'

    def get(self, request, *args, **kwargs):
        attendance = Attendance.objects.filter(employee=request.user)
        profile = Profile.objects.get(pk=self.kwargs.get('pk'))

        datum = request.GET.get('q')
        if datum:
            filter_date = datetime.strptime(datum, '%m/%d/%Y')
            print(filter_date)
            if filter_date:
                attendance = Attendance.objects.filter(date=filter_date, employee__profile__id=profile.pk)
        else:
            attendance = Attendance.objects.filter(employee__profile__id=profile.pk, date=datetime.today())

        return render(request, self.template_name, {'attendance': attendance, 'profile': profile})

def attendance_record(request):
    template_name = "core/attendance_record.html"
    # filter_date = request.GET.get('q')
    datum = request.GET.get('q')
    if datum:
        filter_date = datetime.strptime(datum, '%m/%d/%Y')
        print(filter_date)
        if filter_date:
            attendance = Attendance.objects.filter(date=filter_date)
    else:
        attendance = Attendance.objects.filter(date=datetime.today())


    return render(request, template_name,{'attendance':attendance})

def start_attendance(request):
    a = attendance()
    for i in a:
        user=User.objects.get(username=i[0])
        time=i[1]
        date=i[2]
        Attendance.objects.create(employee=user, time=time, date=date)
    return render(request, 'core/manage_attendance.html')

@transaction.atomic
def update_profile(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST)
        profile_form = ProfileForm(request.POST)
        if user_form.is_valid() and profile_form.is_valid():
            u = user_form.save()
            u.set_password(request.POST['password'])
            u.save()
            Profile.objects.get_or_create(
                user=u,
                position=request.POST.get('position'),
                phone=request.POST.get('phone'),
                address=request.POST.get('address'),
                birth_date=request.POST.get('birth_date'),
                joined_date=request.POST.get('joined_date'),
                image=request.FILES.get('image'))
            # profile_form.save()
            messages.success(request, ('Your profile was successfully updated!'))
            return redirect('core:main_dashboard')
        else:
            messages.error(request, ('Please correct the error below.'))
    else:
        user_form = UserForm()
        profile_form = ProfileForm()
        # user_form = UserForm(instance=request.user)
        # profile_form = ProfileForm(instance=request.user.profile)
    return render(request, 'core/register_new_member.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })






