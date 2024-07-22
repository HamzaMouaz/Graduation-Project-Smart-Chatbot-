
from django.urls import path
from . import views
from chatbotApp import views


urlpatterns = [
    path('', views.index, name='index'),
    path('w', views.non, name='non'),
    path('login/', views.view_login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('signup/', views.signup, name='signup'), # Ajoutez cette ligne
    path('historique/', views.historique, name='historique'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('getResponse/', views.getResponse, name='getResponse'),
    #path('getResponse/', views.HuggingFaceInferenceView, name='getResponse'),

    



    
]

#path("signup/", views.authView, name="authView"),