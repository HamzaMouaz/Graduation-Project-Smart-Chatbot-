from django.db import models
from django.contrib.auth.models import AbstractUser

class Userr(AbstractUser):
    pass

class Conversation(models.Model):
    user = models.ForeignKey(Userr, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)


# Create your models here.
