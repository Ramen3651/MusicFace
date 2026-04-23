import threading
import requests
import json
import io
import webbrowser

import cv2
from deepface import DeepFace
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.utils import platform
from kivy.core.image import Image as CoreImage
from kivy.metrics import dp

from database import (
    register_user, login_user, get_username,
    update_username, update_password,
    log_emotion, get_emotion_counts, get_most_detected_emotion,
    log_recommended_songs, get_seen_songs,
    like_song, unlike_song, get_liked_songs, is_liked
)

# Constants

FLASK_URL    = "http://127.0.0.1:5000/recommend"

SAGE         = (0.52, 0.67, 0.60, 1)      
SAGE_DARK    = (0.40, 0.55, 0.48, 1)
WHITE        = (1, 1, 1, 1)
OFF_WHITE    = (0.97, 0.97, 0.97, 1)
DARK_TEXT    = (0.15, 0.15, 0.15, 1)
MUTED_TEXT   = (0.45, 0.45, 0.45, 1)
HEART_RED    = (0.85, 0.25, 0.25, 1)
HEART_EMPTY  = (0.75, 0.75, 0.75, 1)

MAP_TO_4 = {
    "happy":    "happy",
    "sad":      "sad",
    "angry":    "angry",
    "disgust":  "angry",
    "neutral":  "neutral",
    "fear":     "neutral",
    "surprise": "neutral"
}

EMOTION_LABELS = {
    "happy":   "[Happy]",
    "sad":     "[Sad]",
    "angry":   "[Angry]",
    "neutral": "[Neutral]"
}

EMOTION_COLOURS = {
    "happy":   "#F5C842",
    "sad":     "#4A90D9",
    "angry":   "#E05C5C",
    "neutral": "#85AA96"
}


# Helpers

def open_url(url):
    webbrowser.open(url)


def detect_emotion_from_frames(frames, confidence_threshold=0.5):
    mapped_samples = []
    for frame in frames:
        h, w   = frame.shape[:2]
        x1, y1 = int(w * 0.25), int(h * 0.15)
        x2, y2 = int(w * 0.75), int(h * 0.90)
        cropped = frame[y1:y2, x1:x2]
        rgb     = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        try:
            result        = DeepFace.analyze(rgb, actions=["emotion"],
                                             detector_backend="opencv",
                                             enforce_detection=False)
            emotions_dict = result[0]["emotion"]
            dominant      = result[0]["dominant_emotion"]
            confidence    = emotions_dict[dominant] / 100.0
            if confidence >= confidence_threshold:
                mapped = MAP_TO_4.get(dominant.lower(), "neutral")
                mapped_samples.append(mapped)
        except Exception:
            continue
    if not mapped_samples:
        return "neutral"
    return Counter(mapped_samples).most_common(1)[0][0]


def make_bg(widget, r, g, b, a=1):
    """Draw a solid colour background on any widget."""
    with widget.canvas.before:
        Color(r, g, b, a)
        rect = Rectangle(size=widget.size, pos=widget.pos)
    widget.bind(size=lambda w, v: setattr(rect, "size", v),
                pos=lambda w, v:  setattr(rect, "pos",  v))


def styled_button(text, bg=SAGE, fg=WHITE, bold=False, height=dp(44), font_size=dp(14)):
    btn = Button(
        text=text,
        size_hint_y=None,
        height=height,
        background_normal="",
        background_color=bg,
        color=fg,
        bold=bold,
        font_size=font_size
    )
    return btn


def input_field(hint, password=False):
    return TextInput(
        hint_text=hint,
        password=password,
        multiline=False,
        size_hint_y=None,
        height=dp(42),
        background_color=(0.96, 0.96, 0.96, 1),
        foreground_color=DARK_TEXT,
        padding=[dp(10), dp(10)],
        font_size=dp(14)
    )



# Shared top bar and bottom nav

class TopBar(BoxLayout):
    def __init__(self, username="", on_heart=None, **kwargs):
        super().__init__(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(52),
            padding=[dp(12), dp(8)],
            spacing=dp(8),
            **kwargs
        )
        make_bg(self, *SAGE[:3])

        self.welcome = Label(
            text=f"Welcome, {username}",
            color=WHITE,
            bold=True,
            font_size=dp(15),
            halign="left",
            valign="middle"
        )
        self.welcome.bind(size=self.welcome.setter("text_size"))
        self.add_widget(self.welcome)

        if on_heart:
            heart_btn = Button(
                text="Liked",
                size_hint=(None, None),
                size=(dp(58), dp(36)),
                background_normal="",
                background_color=SAGE_DARK,
                color=WHITE,
                font_size=dp(12),
                bold=True
            )
            heart_btn.bind(on_press=on_heart)
            self.add_widget(heart_btn)

    def set_username(self, username):
        self.welcome.text = f"Welcome, {username}"


class BottomNav(BoxLayout):
    def __init__(self, on_discover, on_analytics, on_profile, **kwargs):
        super().__init__(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            **kwargs
        )
        make_bg(self, *SAGE[:3])

        labels = [("Discover", on_discover), ("Analytics", on_analytics), ("Profile", on_profile)]
        for text, callback in labels:
            btn = Button(
                text=text,
                background_normal="",
                background_color=SAGE,
                color=WHITE,
                font_size=dp(13),
                bold=True
            )
            btn.bind(on_press=callback)
            self.add_widget(btn)

        self._buttons = self.children[:]

    def highlight(self, index):
        for i, btn in enumerate(reversed(self._buttons)):
            btn.background_color = SAGE_DARK if i == index else SAGE


# Login / Signup screen

class AuthScreen(BoxLayout):
    def __init__(self, on_login_success, **kwargs):
        super().__init__(orientation="vertical", padding=dp(32), spacing=dp(16), **kwargs)
        make_bg(self, *WHITE[:3])
        self.on_login_success = on_login_success

        # Logo area
        logo = Label(
            text="MusicFace",
            font_size=dp(32),
            bold=True,
            color=SAGE,
            size_hint_y=None,
            height=dp(60)
        )
        self.add_widget(logo)

        tagline = Label(
            text="Music that matches your mood",
            font_size=dp(13),
            color=MUTED_TEXT,
            size_hint_y=None,
            height=dp(24)
        )
        self.add_widget(tagline)

        self.add_widget(Label(size_hint_y=None, height=dp(16)))

        self.username_input  = input_field("Username")
        self.password_input  = input_field("Password", password=True)
        self.feedback        = Label(text="", color=(0.8, 0.2, 0.2, 1),
                                     size_hint_y=None, height=dp(28),
                                     font_size=dp(13))

        self.add_widget(self.username_input)
        self.add_widget(self.password_input)
        self.add_widget(self.feedback)

        login_btn  = styled_button("Log In",  bg=SAGE,      bold=True)
        signup_btn = styled_button("Sign Up", bg=SAGE_DARK, bold=True)
        login_btn.bind(on_press=self.do_login)
        signup_btn.bind(on_press=self.do_signup)

        self.add_widget(login_btn)
        self.add_widget(signup_btn)
        self.add_widget(Label())  # spacer

    def _get_fields(self):
        return self.username_input.text.strip(), self.password_input.text.strip()

    def do_login(self, *_):
        username, password = self._get_fields()
        if not username or not password:
            self.feedback.text = "Please enter both username and password."
            return
        success, result = login_user(username, password)
        if success:
            self.feedback.text = ""
            self.on_login_success(result)
        else:
            self.feedback.text = result

    def do_signup(self, *_):
        username, password = self._get_fields()
        if not username or not password:
            self.feedback.text = "Please enter both username and password."
            return
        if len(password) < 6:
            self.feedback.text = "Password must be at least 6 characters."
            return
        success, result = register_user(username, password)
        if success:
            self.feedback.text = ""
            self.on_login_success(result)
        else:
            self.feedback.text = result


# Discover page

class DiscoverPage(BoxLayout):
    def __init__(self, user_id, **kwargs):
        super().__init__(orientation="vertical", padding=dp(12), spacing=dp(10), **kwargs)
        make_bg(self, *OFF_WHITE[:3])

        self.user_id       = user_id
        self.cap           = None
        self.latest_frame  = None
        self._preview_evt  = None
        self._countdown    = 0
        self._capturing    = False
        self.current_songs = []
        self._consent_given = False

        # Capture button
        self.capture_btn = styled_button(
            "Capture & Detect Emotion",
            bold=True,
            height=dp(50),
            font_size=dp(15)
        )
        self.capture_btn.bind(on_press=self.on_capture_pressed)
        self.add_widget(self.capture_btn)

        # Status line
        self.status = Label(
            text="Press the button to detect your emotion",
            color=MUTED_TEXT,
            size_hint_y=None,
            height=dp(24),
            font_size=dp(13)
        )
        self.add_widget(self.status)

        # Detected emotion display
        self.emotion_label = Label(
            text="",
            font_size=dp(18),
            bold=True,
            color=SAGE,
            size_hint_y=None,
            height=dp(30)
        )
        self.add_widget(self.emotion_label)

        # Countdown label (only visible during 3-2-1)
        self.countdown_label = Label(
            text="",
            font_size=dp(48),
            bold=True,
            color=SAGE,
            size_hint_y=None,
            height=dp(0)
        )
        self.add_widget(self.countdown_label)

        # Camera preview (zero height when hidden)
        self.camera_feed = Image(size_hint_y=None, height=dp(0), opacity=0)
        self.add_widget(self.camera_feed)

        # Song list — takes all remaining space
        scroll = ScrollView(size_hint_y=1)
        self.list_layout = GridLayout(cols=1, size_hint_y=None, spacing=dp(6), padding=dp(4))
        self.list_layout.bind(minimum_height=self.list_layout.setter("height"))
        scroll.add_widget(self.list_layout)
        self.add_widget(scroll)

    # Camera

    def _open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            Clock.schedule_once(lambda dt: setattr(self.status, "text", "Could not open camera."))
            return
        self.cap = cap
        Clock.schedule_once(lambda dt: self._start_preview())

    def _start_preview(self):
        self.camera_feed.height  = dp(200)
        self.camera_feed.opacity = 1
        self.countdown_label.height = dp(60)
        self._preview_evt = Clock.schedule_interval(self._update_preview, 1 / 20)

    def _update_preview(self, dt):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.latest_frame = frame.copy()
        frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_flipped = cv2.flip(frame_rgb, 0)
        h, w, _       = frame_flipped.shape
        texture       = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(frame_flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.camera_feed.texture = texture

    def _stop_camera(self):
        if self._preview_evt:
            self._preview_evt.cancel()
            self._preview_evt = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_feed.opacity = 0
        self.camera_feed.height  = dp(0)
        self.countdown_label.height = dp(0)


    # Capture flow with countdown
    def on_capture_pressed(self, *_):
        if self._capturing:
            return
        if self._consent_given:
            self._start_capture()
        else:
            self._show_consent_popup()

    def _show_consent_popup(self):
        content = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))

        msg = Label(
            text=(
                "MusicFace would like to access your camera "
                "to detect your emotion.\n\n"
                "No images are saved or stored at any point."
            ),
            color=(1, 1, 1, 1),
            halign="center",
            valign="middle",
            size_hint_y=1,
            font_size=dp(13)
        )
        msg.bind(size=msg.setter("text_size"))
        content.add_widget(msg)

        btn_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(44),
            spacing=dp(8)
        )

        allow_btn = styled_button("Allow", bg=SAGE, bold=True)
        deny_btn  = styled_button("Deny",  bg=(0.75, 0.25, 0.25, 1), bold=True)

        # Use a list to hold the popup reference so the lambdas can access it
        popup_holder = []

        def on_allow(*_):
            popup_holder[0].dismiss()
            self._start_capture()

        def on_deny(*_):
            popup_holder[0].dismiss()

        allow_btn.bind(on_press=on_allow)
        deny_btn.bind(on_press=on_deny)

        btn_row.add_widget(allow_btn)
        btn_row.add_widget(deny_btn)
        content.add_widget(btn_row)

        popup = Popup(
            title="Camera Access",
            content=content,
            size_hint=(0.85, 0.38),
            auto_dismiss=False
        )
        popup_holder.append(popup)
        popup.open()

    def _start_capture(self):
        self._consent_given       = True
        self._capturing           = True
        self.capture_btn.disabled = True
        self.list_layout.clear_widgets()
        self.emotion_label.text   = ""
        self.status.text          = ""
        threading.Thread(target=self._open_camera, daemon=True).start()
        Clock.schedule_once(lambda dt: self._start_countdown(), 0.6)

    def _start_countdown(self):
        self._countdown = 3
        self.countdown_label.text = str(self._countdown)
        Clock.schedule_interval(self._tick_countdown, 1)

    def _tick_countdown(self, dt):
        self._countdown -= 1
        if self._countdown > 0:
            self.countdown_label.text = str(self._countdown)
        else:
            self.countdown_label.text = ""
            self._do_capture()
            return False  # stop interval

    def _do_capture(self):
        self.status.text = "Analysing emotion..."
        frame = self.latest_frame
        if frame is None:
            self.status.text    = "No frame captured. Please try again."
            self._capturing     = False
            self.capture_btn.disabled = False
            self._stop_camera()
            return
        threading.Thread(
            target=self._run_detection,
            args=(frame.copy(),),
            daemon=True
        ).start()

    def _run_detection(self, frame):
        emotion = detect_emotion_from_frames([frame] * 5)
        Clock.schedule_once(lambda dt: self._on_detected(emotion))

    def _on_detected(self, emotion):
        self._stop_camera()
        self._capturing = False
        self.capture_btn.disabled = False

        log_emotion(self.user_id, emotion)

        label_text = EMOTION_LABELS.get(emotion, emotion.capitalize())
        self.emotion_label.text = f"Detected: {label_text}"
        self.status.text        = "Fetching recommendations..."
        self._fetch_songs(emotion)

    # Song recommendations

    def _fetch_songs(self, emotion):
        seen      = list(get_seen_songs(self.user_id))
        seen_list = [{"track_name": t, "artists": a} for t, a in seen]

        try:
            r = requests.get(
                FLASK_URL,
                params={
                    "emotion": emotion,
                    "n":       10,
                    "seen":    json.dumps(seen_list)
                },
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.status, "text", f"Error: {e}"))
            return

        songs = data.get("songs", [])
        if songs:
            log_recommended_songs(self.user_id, songs)

        Clock.schedule_once(lambda dt: self._display_songs(songs))

    def _display_songs(self, songs):
        self.current_songs = songs
        self.list_layout.clear_widgets()
        self.status.text = f"{len(songs)} songs recommended" if songs else "No songs found."

        for s in songs:
            self._add_song_row(s)

    def _add_song_row(self, s):
        track   = s["track_name"]
        artists = s["artists"]
        score   = int(s.get("match_score", 0) * 100)

        row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(52),
            spacing=dp(6),
            padding=[dp(4), dp(4)]
        )
        make_bg(row, 1, 1, 1)

        # Song button (opens YouTube)
        search_query = f"{track} {artists} official".replace(" ", "+")
        youtube_url  = f"https://www.youtube.com/results?search_query={search_query}"

        song_btn = Button(
            text=f"{track} — {artists}  ({score}% match)",
            halign="left",
            valign="middle",
            background_normal="",
            background_color=WHITE,
            color=DARK_TEXT,
            font_size=dp(13)
        )
        song_btn.bind(size=song_btn.setter("text_size"))
        song_btn.bind(on_press=lambda inst, url=youtube_url: open_url(url))

        # Heart button
        already_liked = is_liked(self.user_id, track, artists)
        heart_btn = Button(
            text="Liked" if already_liked else "Like",
            size_hint=(None, None),
            size=(dp(54), dp(44)),
            background_normal="",
            background_color=SAGE if already_liked else (0.88, 0.88, 0.88, 1),
            color=WHITE,
            font_size=dp(11),
            bold=True
        )

        def toggle_like(inst, t=track, a=artists, hb=heart_btn):
            if hb.text == "Liked":
                unlike_song(self.user_id, t, a)
                hb.text = "Like"
                hb.background_color = (0.88, 0.88, 0.88, 1)
            else:
                like_song(self.user_id, t, a)
                hb.text = "Liked"
                hb.background_color = SAGE

        heart_btn.bind(on_press=toggle_like)

        row.add_widget(song_btn)
        row.add_widget(heart_btn)
        self.list_layout.add_widget(row)

    def show_liked_songs(self, *_):
        liked = get_liked_songs(self.user_id)

        # Outer layout: scroll takes all space, close button pinned at bottom
        content = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))

        if not liked:
            content.add_widget(Label(
                text="No liked songs yet.",
                color=DARK_TEXT,
                size_hint_y=1
            ))
        else:
            # ScrollView must have size_hint_y=1 so it fills the popup space
            scroll = ScrollView(size_hint_y=1)
            grid   = GridLayout(cols=1, size_hint_y=None, spacing=dp(4), padding=[dp(4), dp(4)])
            grid.bind(minimum_height=grid.setter("height"))

            for s in liked:
                row = BoxLayout(
                    orientation="horizontal",
                    size_hint_y=None,
                    height=dp(48),
                    padding=[dp(8), dp(4)],
                    spacing=dp(8)
                )
                make_bg(row, 1, 1, 1)

                lbl = Label(
                    text=f"{s['track_name']}  —  {s['artists']}",
                    color=DARK_TEXT,
                    halign="left",
                    valign="middle",
                    font_size=dp(13)
                )
                lbl.bind(size=lbl.setter("text_size"))

                liked_badge = Label(
                    text="Liked",
                    color=SAGE,
                    bold=True,
                    size_hint=(None, None),
                    size=(dp(44), dp(28)),
                    font_size=dp(11)
                )

                row.add_widget(lbl)
                row.add_widget(liked_badge)
                grid.add_widget(row)

            scroll.add_widget(grid)
            content.add_widget(scroll)

        close_btn = styled_button("Close", height=dp(44))
        close_btn.size_hint_y = None
        close_btn.bind(on_press=lambda *_: popup.dismiss())
        content.add_widget(close_btn)

        popup = Popup(
            title="Liked Songs",
            content=content,
            size_hint=(0.9, 0.78)
        )
        popup.open()


# Analytics page

class AnalyticsPage(BoxLayout):
    def __init__(self, user_id, **kwargs):
        super().__init__(orientation="vertical", padding=dp(16), spacing=dp(12), **kwargs)
        make_bg(self, *OFF_WHITE[:3])
        self.user_id = user_id

        self.top_label = Label(
            text="No emotion data yet.",
            font_size=dp(18),
            bold=True,
            color=SAGE,
            size_hint_y=None,
            height=dp(40)
        )
        self.add_widget(self.top_label)

        self.chart_img = Image(size_hint_y=1)
        self.add_widget(self.chart_img)

    def refresh(self):
        counts = get_emotion_counts(self.user_id)

        if not counts:
            self.top_label.text    = "No emotion data yet."
            self.chart_img.texture = None
            return

        # Find the highest count then collect all emotions that share it (tie handling)
        max_count = max(counts.values())
        top_emotions = [e for e, c in counts.items() if c == max_count]
        top_labels   = [EMOTION_LABELS.get(e, e.capitalize()) for e in top_emotions]

        if len(top_labels) == 1:
            self.top_label.text = f"Most detected: {top_labels[0]}"
        else:
            joined = "  &  ".join(top_labels)
            self.top_label.text = f"Most detected (tied): {joined}"

        # Build pie chart with matplotlib
        labels = list(counts.keys())
        sizes  = list(counts.values())
        colours = [EMOTION_COLOURS.get(e, "#AAAAAA") for e in labels]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.pie(
            sizes,
            labels=labels,
            colors=colours,
            autopct="%1.0f%%",
            startangle=140,
            textprops={"fontsize": 12}
        )
        ax.set_title("Emotion Detection History", fontsize=13, pad=12)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        core_img = CoreImage(buf, ext="png")
        self.chart_img.texture = core_img.texture


# Profile page

class ProfilePage(BoxLayout):
    def __init__(self, user_id, on_logout, on_username_changed, **kwargs):
        super().__init__(orientation="vertical", padding=dp(24), spacing=dp(14), **kwargs)
        make_bg(self, *OFF_WHITE[:3])

        self.user_id            = user_id
        self.on_logout          = on_logout
        self.on_username_changed = on_username_changed

        self.username_label = Label(
            text=get_username(user_id),
            font_size=dp(22),
            bold=True,
            color=SAGE,
            size_hint_y=None,
            height=dp(48)
        )
        self.add_widget(self.username_label)

        divider = Label(
            text="Account Settings",
            font_size=dp(13),
            color=MUTED_TEXT,
            size_hint_y=None,
            height=dp(28)
        )
        self.add_widget(divider)

        # Change username
        self.new_username = input_field("New username")
        change_user_btn   = styled_button("Change Username")
        change_user_btn.bind(on_press=self.do_change_username)

        # Change password
        self.new_password  = input_field("New password (min 6 chars)", password=True)
        change_pass_btn    = styled_button("Change Password")
        change_pass_btn.bind(on_press=self.do_change_password)

        self.feedback = Label(
            text="",
            color=(0.2, 0.6, 0.3, 1),
            size_hint_y=None,
            height=dp(28),
            font_size=dp(13)
        )

        logout_btn = styled_button("Log Out", bg=(0.75, 0.25, 0.25, 1))
        logout_btn.bind(on_press=lambda *_: self.on_logout())

        for w in [self.new_username, change_user_btn,
                  self.new_password, change_pass_btn,
                  self.feedback, logout_btn]:
            self.add_widget(w)

        self.add_widget(Label())  # spacer

    def refresh(self):
        self.username_label.text = get_username(self.user_id)

    def do_change_username(self, *_):
        new = self.new_username.text.strip()
        if not new:
            self.feedback.text  = "Please enter a new username."
            self.feedback.color = (0.8, 0.2, 0.2, 1)
            return
        success, msg = update_username(self.user_id, new)
        self.feedback.color = (0.2, 0.6, 0.3, 1) if success else (0.8, 0.2, 0.2, 1)
        self.feedback.text  = msg
        if success:
            self.username_label.text = new
            self.new_username.text   = ""
            self.on_username_changed(new)

    def do_change_password(self, *_):
        new = self.new_password.text.strip()
        if len(new) < 6:
            self.feedback.color = (0.8, 0.2, 0.2, 1)
            self.feedback.text  = "Password must be at least 6 characters."
            return
        success, msg         = update_password(self.user_id, new)
        self.feedback.color  = (0.2, 0.6, 0.3, 1) if success else (0.8, 0.2, 0.2, 1)
        self.feedback.text   = msg
        self.new_password.text = ""


# Main app screen (post-login shell with top bar, pages, bottom nav)

class MainScreen(BoxLayout):
    def __init__(self, user_id, on_logout, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        make_bg(self, *WHITE[:3])

        self.user_id   = user_id
        self.on_logout = on_logout

        # Pages
        self.discover  = DiscoverPage(user_id)
        self.analytics = AnalyticsPage(user_id)
        self.profile   = ProfilePage(
            user_id,
            on_logout=self._do_logout,
            on_username_changed=self._on_username_changed
        )

        # Top bar
        self.top_bar = TopBar(
            username=get_username(user_id),
            on_heart=self.discover.show_liked_songs
        )
        self.add_widget(self.top_bar)

        # Page container
        self.page_container = BoxLayout(orientation="vertical")
        self.page_container.add_widget(self.discover)
        self.add_widget(self.page_container)

        # Bottom nav
        self.nav = BottomNav(
            on_discover=lambda *_: self._switch_page(0),
            on_analytics=lambda *_: self._switch_page(1),
            on_profile=lambda *_: self._switch_page(2)
        )
        self.add_widget(self.nav)

        self.current_page = 0
        self.nav.highlight(0)

    def _switch_page(self, index):
        self.page_container.clear_widgets()
        pages = [self.discover, self.analytics, self.profile]
        self.page_container.add_widget(pages[index])
        self.current_page = index
        self.nav.highlight(index)

        if index == 1:
            self.analytics.refresh()
        if index == 2:
            self.profile.refresh()

    def _on_username_changed(self, new_username):
        self.top_bar.set_username(new_username)

    def _do_logout(self):
        self.on_logout()



# Root widget — switches between auth and main

class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._show_auth()

    def _show_auth(self):
        self.clear_widgets()
        auth = AuthScreen(on_login_success=self._on_login)
        self.add_widget(auth)

    def _on_login(self, user_id):
        self.clear_widgets()
        main = MainScreen(user_id=user_id, on_logout=self._show_auth)
        self.add_widget(main)


class MusicFaceApp(App):
    def build(self):
        return RootWidget()


if __name__ == "__main__":
    MusicFaceApp().run()