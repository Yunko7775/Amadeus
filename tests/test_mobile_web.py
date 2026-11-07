import re
import unittest
from unittest.mock import patch

from shared import config

config.API_KEY = "test-key"

from backend import flask_backend


LAN_CLIENT = {"REMOTE_ADDR": "192.168.1.25"}


class MobileWebTests(unittest.TestCase):
    def test_remote_chat_page_requires_access_code(self):
        with patch.object(flask_backend, "WEB_ACCESS_CODE", "a" * 24):
            with flask_backend.app.test_client() as client:
                page = client.get("/", environ_base=LAN_CLIENT)
                api = client.post(
                    "/ask",
                    json={"user_id": "mobile", "message": "hello"},
                    environ_base=LAN_CLIENT,
                )

        self.assertEqual(page.status_code, 302)
        self.assertEqual(page.headers["Location"], "/login")
        self.assertEqual(api.status_code, 401)
        self.assertEqual(api.get_json()["error"], "authentication_required")

    def test_login_grants_access_but_requires_csrf_for_chat_posts(self):
        with patch.object(flask_backend, "WEB_ACCESS_CODE", "a" * 24):
            with flask_backend.app.test_client() as client:
                login_page = client.get("/login", environ_base=LAN_CLIENT)
                csrf_token = re.search(
                    r'name="csrf_token" value="([^"]+)"',
                    login_page.get_data(as_text=True),
                ).group(1)
                login = client.post(
                    "/login",
                    data={
                        "csrf_token": csrf_token,
                        "access_code": "a" * 24,
                    },
                    environ_base=LAN_CLIENT,
                )
                page = client.get("/", environ_base=LAN_CLIENT)
                chat = client.post(
                    "/ask",
                    json={"user_id": "mobile", "message": "hello"},
                    environ_base=LAN_CLIENT,
                )

        self.assertEqual(login.status_code, 302)
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="chat-form"', page.get_data(as_text=True))
        self.assertEqual(chat.status_code, 400)
        self.assertEqual(chat.get_json()["error"], "csrf_token_invalid")

    def test_invalid_access_code_is_rejected(self):
        with patch.object(flask_backend, "WEB_ACCESS_CODE", "a" * 24):
            with flask_backend.app.test_client() as client:
                login_page = client.get("/login", environ_base=LAN_CLIENT)
                csrf_token = re.search(
                    r'name="csrf_token" value="([^"]+)"',
                    login_page.get_data(as_text=True),
                ).group(1)
                response = client.post(
                    "/login",
                    data={"csrf_token": csrf_token, "access_code": "wrong"},
                    environ_base=LAN_CLIENT,
                )

        self.assertEqual(response.status_code, 401)
        self.assertIn("存取碼不正確", response.get_data(as_text=True))

    def test_remote_access_stays_disabled_without_code(self):
        with patch.object(flask_backend, "WEB_ACCESS_CODE", ""):
            with flask_backend.app.test_client() as client:
                remote = client.get("/", environ_base=LAN_CLIENT)
                local = client.get("/", environ_base={"REMOTE_ADDR": "127.0.0.1"})

        self.assertEqual(remote.status_code, 403)
        self.assertEqual(remote.get_json()["error"], "lan_access_disabled")
        self.assertEqual(local.status_code, 200)

    def test_mobile_page_has_live2d_canvas_and_voice_support(self):
        with flask_backend.app.test_client() as client:
            response = client.get("/")
            mobile_script = client.get("/static/mobile.js")
            page = response.get_data(as_text=True)
            script_data = mobile_script.get_data()
            response.close()
            mobile_script.close()
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="live2d-canvas"', page)
        self.assertIn("l2d@2.1.1", page)
        self.assertIn("聽日語語音".encode("utf-8"), script_data)

    def test_model_assets_are_served_for_the_authenticated_web_client(self):
        with patch.object(flask_backend, "WEB_ACCESS_CODE", "a" * 24):
            with flask_backend.app.test_client() as client:
                page = client.get("/models/kurisu/kurisu.model.json", environ_base=LAN_CLIENT)
                login_page = client.get("/login", environ_base=LAN_CLIENT)
                csrf_token = re.search(
                    r'name="csrf_token" value="([^"]+)"',
                    login_page.get_data(as_text=True),
                ).group(1)
                client.post(
                    "/login",
                    data={
                        "csrf_token": csrf_token,
                        "access_code": "a" * 24,
                    },
                    environ_base=LAN_CLIENT,
                )
                assets = {
                    "kurisu.model.json",
                    "kurisu.moc",
                    "kurisu.physics.json",
                    "kurisu.2048/texture_00.png",
                }
                responses = {
                    name: client.get(
                        f"/models/kurisu/{name}",
                        environ_base=LAN_CLIENT,
                    )
                    for name in assets
                }
                model_data = responses["kurisu.model.json"].get_data()
                asset_data = {
                    name: response.get_data()
                    for name, response in responses.items()
                }
                for response in responses.values():
                    response.close()

        self.assertEqual(page.status_code, 302)
        self.assertIn(b'"kurisu.moc"', model_data)
        for name in assets:
            with self.subTest(asset=name):
                self.assertEqual(responses[name].status_code, 200)
                self.assertTrue(asset_data[name])
        self.assertTrue(asset_data["kurisu.2048/texture_00.png"].startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
