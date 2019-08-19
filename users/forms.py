from django import forms
from django.contrib.auth.models import User
from .models import Profile
from django.forms import ModelForm, inlineformset_factory

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'username', 'password')

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('position', 'phone', 'address', 'birth_date', 'joined_date', 'image')