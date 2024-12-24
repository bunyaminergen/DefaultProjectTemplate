import os
import yaml


class PromptManager:
    """
    Prompt'ları yöneten sınıf. Config dizinindeki prompts.yaml dosyasını yükler.
    """

    def __init__(self, config_path=None):
        """
        PromptManager örneği oluşturur.
        :param config_path: YAML dosyasının yolu. Varsayılan olarak config/prompts.yaml kullanılır.
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../config/prompts.yaml"
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        """
        Config dizinindeki YAML dosyasını yükler ve içeriğini döner.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"YAML dosyası bulunamadı: {self.config_path}")
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get_prompt(self, prompt_name, **kwargs):
        """
        Belirli bir prompt'u al ve dinamik değişkenlerle formatla.
        :param prompt_name: Prompt'un adı (YAML dosyasındaki anahtar).
        :param kwargs: Dinamik değerler.
        :return: Formatlanmış prompt.
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' bulunamadı.")

        prompt = self.prompts[prompt_name]
        return {key: value.format(**kwargs) if isinstance(value, str) else value
                for key, value in prompt.items()}


# PromptManager örneği oluşturma
prompt_manager = PromptManager()
