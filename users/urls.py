from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    path('user-dashboard/', views.MainDashboard.as_view(), name='user_dashboard'),
    path('user-profile/<int:pk>', views.UserProfile.as_view(), name='user_profile'),
    path('attendance-record', views.attendance_record, name='attendance_record'),
    path('about/', views.About.as_view(), name='about'),
    path('contact/', views.Contact.as_view(), name='contact'),
]
