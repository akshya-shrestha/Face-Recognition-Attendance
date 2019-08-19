from django.db import models
from django.forms import ModelForm
from django.contrib.auth.models import User

# Create your models here.

class Attendance(models.Model):
    employee = models.ForeignKey(User, on_delete=models.CASCADE)
    time = models.CharField(max_length=50, null=True)
    date = models.DateField('attendance date')

    def __str__(self):
        return self.employee.username


