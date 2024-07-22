from django.contrib import admin
from .models import Userr,Conversation

# Enregistrement des modèles auprès de l'interface d'administration
admin.site.register(Userr)
admin.site.register(Conversation)

