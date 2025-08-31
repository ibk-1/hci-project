from django.db import models

class Movie(models.Model):
    ml_id = models.IntegerField(unique=True)  # MovieLens movieId
    title = models.CharField(max_length=255)
    year = models.IntegerField(null=True, blank=True)
    genres = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.title


class UserSession(models.Model):
    session_key = models.CharField(max_length=64, unique=True)
    created = models.DateTimeField(auto_now_add=True)


class Interaction(models.Model):
    user = models.ForeignKey(UserSession, on_delete=models.CASCADE, related_name="interactions")
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.FloatField()  # 0.5–5.0
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "movie")
