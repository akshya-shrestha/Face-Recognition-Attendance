from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.generic import TemplateView, DetailView
from users.models import Profile
from core.models import Attendance
from datetime import datetime
from django.urls import reverse
from django.core.exceptions import PermissionDenied


# Create your views here.
class MainDashboard(TemplateView):
    template_name = "users/main_dashboard.html"

class About(TemplateView):
    template_name = 'users/about.html'

class Contact(TemplateView):
    template_name = 'users/contact.html'

class UserProfile(DetailView):
    template_name = 'users/profile.html'
    model = Profile
    context_object_name = 'profile'
    def dispatch(self, request, *args, **kwargs):
        if request.user.profile.pk == self.kwargs['pk']:
            return super(UserProfile, self).dispatch(request, *args, **kwargs)
        else:
            raise PermissionDenied

def attendance_record(request):
    template_name = "users/attendance_record.html"
    # import ipdb; ipdb.set_trace()
    attendance = Attendance.objects.filter(employee=request.user)
    datum = request.GET.get('q')
    if datum:
        filter_date = datetime.strptime(datum, '%m/%d/%Y')
        print(filter_date)
        if filter_date:
            attendance = Attendance.objects.filter(date=filter_date, employee=request.user)
    else:
        attendance = Attendance.objects.filter(employee=request.user, date=datetime.today())

    return render(request, template_name, {'attendance': attendance})