from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.MainDashboard.as_view(), name='main_dashboard'),
    path('bod/', views.BOD.as_view(), name='bod'),
    path('about/', views.About.as_view(), name='about'),
    path('contact/', views.Contact.as_view(), name='contact'),
    path('manage-attendance/', views.ManageAttendance.as_view(), name='manage_attendance'),
    path('employee-record/', views.EmployeeRecord.as_view(), name='employee_record'),
    path('attendance-record', views.attendance_record, name='attendance_record'),
    # path('individual-attendance-record', views.individual_attendance, name='individual_attendance_record'),
    path('register-new-member', views.update_profile, name='register_new_member'),
    path('user-profile/<int:pk>', views.UserProfile.as_view(), name='user_profile'),
    path('delete-user/<int:pk>', views.DeleteEmployee.as_view(), name='delete_user'),
    path('update-user/<int:pk>', views.UpdateEmployee.as_view(), name='update_user'),
    path('start-attendance', views.start_attendance, name='start_attendance'),
]