"""
Multi-Modal Visual Inspection Engine (VLM Client & Glare Classifier)
Part of irdpl_toolpath_verifier package.
"""

import json
import urllib.request
import urllib.error
import numpy as np


class VisualInspector:
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "qwen2.5vl:3b",
        timeout_seconds: float = 5.0
    ):
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout_seconds

        self.system_prompt = (
            "You are an industrial computer vision quality inspector in a robotic coating cell. "
            "A 3D depth sensor lost tracking over this steel workpiece region. "
            "Determine: Is this an optical SPECULAR_GLARE reflection on continuous solid metal, "
            "or is there an actual PHYSICAL_VOID, cut, or crack? "
            "Respond ONLY with raw JSON: "
            '{"classification": "SPECULAR_GLARE" or "PHYSICAL_VOID", "surface_continuous": true or false, "confidence": float}'
        )

    def is_service_online(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def classify_surface_patch(self, image_bytes_base64: str) -> dict:
        """
        Sends image crop to local VLM and extracts structured classification.
        Includes graceful fallback if VLM is offline.
        """
        default_fallback = {
            "classification": "SPECULAR_GLARE",
            "surface_continuous": True,
            "confidence": 0.50,
            "vlm_online": False,
            "reason": "VLM offline; defaulted to geometric verification."
        }

        if not image_bytes_base64 or not self.is_service_online():
            return default_fallback

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": self.system_prompt,
                    "images": [image_bytes_base64]
                }
            ],
            "options": {"temperature": 0.0},
            "stream": False
        }

        try:
            req_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.ollama_host}/api/chat",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                raw_text = result.get('message', {}).get('content', '').strip()

                # Clean markdown code blocks if returned
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                parsed = json.loads(raw_text)
                parsed["vlm_online"] = True
                return parsed

        except Exception as e:
            fallback = dict(default_fallback)
            fallback["reason"] = f"VLM Query Exception: {e}"
            return fallback
