class Dog2:
    country = "世界"
    voice = "国によって変わる"

    def __init__(self, name):
        self.name = name

    def bark(self):
        print(self.voice)

    @classmethod
    def description(self):
        print(f"{self.country}の犬の鳴き声は{self.voice}")

class English_Dog2(Dog2):
    country = "英語圏"
    voice = "bow!"

    def __init__(self, name = "Buddy"):
        super().__init__(name)

class Japanese_Dog2(Dog2):
    country = "日本"
    voice = "ワン！"

    def __init__(self, name = "ポチ"):
        super().__init__(name)
