from django import forms

#from chatbotApp.models import CustomUser
from django.contrib.auth.forms import UserCreationForm

from chatbotApp.models import Userr

class SignUpForm(UserCreationForm):
    class Meta:
        model = Userr
        fields = ('username', 'password1', 'password2')

class LoginForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)


