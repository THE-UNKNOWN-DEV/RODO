# RODO
ORBI is an AI Powered, voice recognizing chatbot, which can interact users through speech. It can  help you with tasks , answer questions and keep a smart home gadgets running smoothly. It can be woken up by the phrase “Hey ORBI.” ORBI can even run programs and tasks on the Pi, and also has potential applications in Home Automation.



Code-
ORBI - Raspberry Pi voice assistant (hands-free, like a home Alexa)

How it works:
  1. ORBI listens all the time. Say "Hey ORBI" (you can say your question in the same breath).
  2. It beeps, records until you stop talking, then answers out loud.
  3. After answering it beeps again and keeps listening for a few seconds,
     so you can ask a follow-up without saying "Hey ORBI" again. Say "thank you" or "stop" to end.

Brain:   Groq gpt-oss + free DuckDuckGo web search -> Groq backup model -> local Ollama (if internet is down)
Ears:    Groq Whisper speech-to-text
Mouth:   gTTS + mpg123

Plugins (answered instantly on the Pi, without asking the AI):
  - Time and date        "what time is it", "what's the date"
  - Timers               "set a timer for 5 minutes", "cancel the timer"
  - Reminders            "remind me in 10 minutes to drink water"
  - Weather (free)       "what's the weather", "weather in Mysore", "will it rain tomorrow"
  - Shopping/to-do list  "add milk to my list", "what's on my list", "remove milk from my list", "clear my list"
  - Music from YouTube   "play believer by imagine dragons", "stop music"
  - Volume               "volume up", "volume down", "set volume to 60", "mute", "unmute"
  - Pi health            "how hot is the pi"
  - Help                 "what can you do"

One-time setup (in the Pi Terminal):
    pip install gtts requests yt-dlp ddgs --break-system-packages
    sudo apt install -y mpg123 mpv alsa-utils
To start ORBI automatically at power-on (no screen needed): bash setup_autostart.sh
"""

import os
import re
import json
import sys
import time
import wave
import array
import queue
import threading
import collections
import subprocess
import requests
from gtts import gTTS

try:
    import yt_dlp  # for music. If missing, everything else still works
except ImportError:
    yt_dlp = None

try:  # free DuckDuckGo web search, no key needed
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ---------------- SETTINGS ----------------
HOME_CITY = "Your_City"                  # your city, used for weather. Change it to your city
CHAT_MODEL = "openai/gpt-oss-20b"        # main brain. It searches the web by itself when a question needs it
BACKUP_MODEL = "openai/gpt-oss-120b"     # used if the main model is busy or over its limit
STT_MODEL = "whisper-large-v3-turbo"     # speech-to-text model
USE_MIC = True                           # True = hands-free voice. False = type to ORBI (for testing in Thonny)
MIC_DEVICE = "plughw:CARD=Device,DEV=0"  # your USB mic. Use "" for the default mic, or "plughw:3,0"
MIC_SENSITIVITY = 300                    # sounds quieter than this are ignored. Lower it if ORBI misses your voice
SILENCE_SECONDS = 1.0                    # ORBI stops recording after this much silence
FOLLOW_UP_SECONDS = 6                    # after answering, keep listening this long for another question
LOCAL_MODEL = "llama3.2:1b"              # your existing Ollama model, used only as backup
MUSIC_VOLUME = 70                        # music volume (0-100), a bit lower so you can hear ORBI over it

FOLDER = os.path.dirname(os.path.abspath(__file__))
LIST_FILE = os.path.join(FOLDER, "orbi_list.txt")
KEY_FILE = os.path.join(FOLDER, "orbi_key.txt")


def load_key():
    """The key is kept in orbi_key.txt, so you never have to paste it into the code again,
    even when you download a new version of orbi.py."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
        if key.startswith("gsk_"):
            return key
    try:
        key = input("Paste your Groq API key and press Enter: ").strip().strip('"').strip("'")
    except EOFError:  # running at power-on with no keyboard
        print("[No Groq key found. Run orbi.py once in Thonny to enter it]")
        sys.exit(1)
    with open(KEY_FILE, "w") as f:
        f.write(key)
    print("[Key saved in orbi_key.txt. To change it later, delete that file and run again]")
    return key


GROQ_API_KEY = load_key()

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}"}

history = []                 # remembers the conversation so follow-up questions work
speak_lock = threading.Lock()  # stops two voices (e.g. a timer and a reply) talking over each other
timers = []                  # running timers and reminders
music_process = None         # the music player, while a song is playing
mic = None                   # the always-on microphone (voice mode only)


def system_prompt():
    today = time.strftime("%A, %d %B %Y, %I:%M %p")
    return (
        f"You are ORBI, a friendly home voice assistant running on a Raspberry Pi in India. Today is {today}. "
        "Your own knowledge is old. For news, people in positions (president, prime minister, chief minister, CEO), "
        "prices, scores, results, releases or anything that may have changed, ALWAYS use the web_search tool first "
        "and answer from its results. Do not search for greetings, small talk, maths or general knowledge that never changes. "
        "If you are not sure, say so instead of guessing. "
        "Your replies are spoken aloud, so answer in 1 or 2 short, natural sentences "
        "with no lists, links, symbols or emojis."
    )


# =====================================================================
#                             MOUTH
# =====================================================================
def speak(text):
    print(f"ORBI: {text}")
    with speak_lock:
        if mic:
            mic.pause()   # stop listening while ORBI talks, so it doesn't hear itself
        try:
            voice_file = os.path.join(FOLDER, "response.mp3")
            gTTS(text=text, lang="en", tld="co.in").save(voice_file)  # tld="com" for US accent
            subprocess.run(["mpg123", "-q", voice_file])  # waits until speech ends
        except Exception as e:
            print(f"Voice error: {e}")
        finally:
            if mic:
                mic.resume()


# =====================================================================
#                             EARS
# =====================================================================
BEEP_FILE = os.path.join(FOLDER, "beep.wav")
DONE_FILE = os.path.join(FOLDER, "done.wav")
RATE = 16000
CHUNK = 480                  # 30 milliseconds of sound per chunk
CHUNK_SECONDS = CHUNK / RATE

# Whisper sometimes "hears" these words in silence or background noise, so they are ignored
FAKE_PHRASES = {"", "you", "thank you for watching", "thanks for watching", "uh", "um", "hmm",
                "subtitles by the amara.org community", "please subscribe"}

# Ways Whisper may write "Hey ORBI"
WAKE_WORD = re.compile(
    r"\b(?:(?:hey|hi|hello|ok|okay|a|hay)[\s,]+(?:orbit|or be|orb e|obi|orbe|aubie|ruby|aubrey|all be)"
    r"|orbi|orby|orbie|orbee|orbii)\b[\s,.!?]*")

END_WORDS = {"stop", "thank you", "thanks", "thank you orbi", "thanks orbi", "that's all", "that is all",
             "nothing", "no", "no thanks", "goodbye", "bye", "cancel", "never mind", "nevermind"}


def make_tone(path, freq, seconds):
    """Creates a short beep sound file once (no download needed)."""
    if os.path.exists(path):
        return
    import math, struct
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        n = int(RATE * seconds)
        for i in range(n):
            fade = min(1, i / 400, (n - i) / 400)   # soft start and end, no clicks
            w.writeframes(struct.pack("<h", int(7000 * fade * math.sin(2 * math.pi * freq * i / RATE))))


def play_tone(path, freq, seconds):
    try:
        make_tone(path, freq, seconds)
        if mic:
            mic.pause()
        subprocess.run(["aplay", "-q", path])
    except Exception:
        pass
    finally:
        if mic:
            mic.resume()


def beep():        # "I'm listening"
    play_tone(BEEP_FILE, 880, 0.15)


def done_beep():   # "I've stopped listening"
    play_tone(DONE_FILE, 440, 0.15)


def level(data):
    """How loud a piece of sound is. Near 0 means silence."""
    samples = array.array("h", data)
    return sum(abs(s) for s in samples) / max(len(samples), 1)


class Microphone:
    """Keeps the USB mic recording all the time in the background."""

    def __init__(self):
        self.chunks = queue.Queue()
        self.paused = False
        self.noise = 100.0          # background noise level, learned automatically
        self.proc = None
        self.start()

    def start(self):
        cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "1"]
        if MIC_DEVICE:
            cmd += ["-D", MIC_DEVICE]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        while True:
            data = proc.stdout.read(CHUNK * 2)
            if not data:
                break                       # mic stopped (unplugged?)
            if not self.paused:
                self.chunks.put(data)

    def alive(self):
        return self.proc.poll() is None

    def restart(self):
        err = self.proc.stderr.read().decode(errors="ignore").strip() if self.proc.stderr else ""
        print(f"[Mic stopped working: {err or 'unknown reason'}. Retrying in 5 seconds...]")
        print("[Check the mic is plugged in, or MIC_DEVICE in SETTINGS. Is Thonny also running ORBI?]")
        time.sleep(5)
        self.start()

    def pause(self):
        self.paused = True

    def resume(self):
        time.sleep(0.2)                     # let the speaker's echo die down
        self.clear()
        self.paused = False

    def clear(self):
        while not self.chunks.empty():
            try:
                self.chunks.get_nowait()
            except queue.Empty:
                break

    def calibrate(self):
        """Listens to the room for 1 second to learn how noisy it is."""
        self.clear()
        levels = []
        while len(levels) < int(1 / CHUNK_SECONDS):
            try:
                levels.append(level(self.chunks.get(timeout=3)))
            except queue.Empty:
                if not self.alive():
                    self.restart()
        self.noise = sum(levels) / len(levels)
        print(f"[Room noise level: {self.noise:.0f}. Speech must be louder than {self.threshold():.0f}]")

    def threshold(self):
        return max(MIC_SENSITIVITY, self.noise * 2.5)

    def record(self, wait_seconds=None, max_seconds=15):
        """Waits for someone to talk, records until they stop, and returns the sound.
        Returns None if nobody spoke within wait_seconds."""
        before = collections.deque(maxlen=int(0.4 / CHUNK_SECONDS))  # keep the start of the first word
        frames, started, loud_run, quiet_run = [], False, 0, 0
        start_time = time.time()
        while True:
            if wait_seconds and not started and time.time() - start_time > wait_seconds:
                return None
            try:
                data = self.chunks.get(timeout=1)
            except queue.Empty:
                if not self.alive():
                    self.restart()
                continue
            loud = level(data) > self.threshold()
            if not started:
                before.append(data)
                if loud:
                    loud_run += 1
                    if loud_run >= 3:       # 0.1 second of sound = someone is talking
                        started, frames = True, list(before)
                else:
                    loud_run = 0
                    self.noise = 0.98 * self.noise + 0.02 * level(data)   # slowly re-learn the room
            else:
                frames.append(data)
                quiet_run = 0 if loud else quiet_run + 1
                if quiet_run * CHUNK_SECONDS >= SILENCE_SECONDS:
                    break
                if len(frames) * CHUNK_SECONDS >= max_seconds:
                    break
        return b"".join(frames)


def transcribe(pcm, hint="Hey Orbi."):
    """Sends the recorded sound to Groq Whisper and returns the words."""
    wav = os.path.join(FOLDER, "input.wav")
    with wave.open(wav, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    try:
        with open(wav, "rb") as f:
            r = requests.post(GROQ_STT_URL, headers=HEADERS,
                              files={"file": ("input.wav", f, "audio/wav")},
                              data={"model": STT_MODEL, "language": "en", "temperature": "0",
                                    "prompt": hint}, timeout=30)
        if r.status_code == 429:
            print("[Speech-to-text limit reached for now, waiting a bit]")
            time.sleep(20)
            return ""
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        return "" if clean(text) in FAKE_PHRASES else text
    except Exception as e:
        print(f"[Speech-to-text error: {e}]")
        return ""


def clean(text):
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


def find_wake_word(text):
    """Returns (True, rest of sentence) if "Hey ORBI" was said."""
    lower = text.lower()
    match = WAKE_WORD.search(lower)
    if not match:
        return False, ""
    return True, text[match.end():].strip(" ,.!?")


# =====================================================================
#                             BRAIN (AI)
# =====================================================================
SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for current, up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
}]


def web_search(query):
    """Free DuckDuckGo search. Returns short text snippets for the AI to read."""
    print(f"[Searching the web: {query}]")
    if DDGS is None:
        return "Web search is not installed. Run: pip install ddgs --break-system-packages"
    search = DDGS()
    lines = []
    for kind, count in (("news", 3), ("text", 4)):   # latest news first, then normal web pages
        try:
            for item in getattr(search, kind)(query, max_results=count) or []:
                lines.append(f"- {item.get('title', '')} {item.get('date', '')}: {item.get('body', '')}")
        except Exception as e:
            print(f"[Search {kind} error: {e}]")
    return "\n".join(lines) or "The search found nothing."


def call_groq(messages, model, use_tools):
    payload = {"model": model, "messages": messages, "max_tokens": 1000,
               "temperature": 0.5, "reasoning_effort": "low"}
    if use_tools:
        payload["tools"] = SEARCH_TOOL
        payload["tool_choice"] = "auto"
    r = requests.post(GROQ_CHAT_URL, headers=HEADERS, json=payload, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("rate limit")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def ask_groq(messages, model):
    messages = list(messages)
    for _ in range(3):  # the AI may search up to 2 times, then must answer
        msg = call_groq(messages, model, use_tools=bool(DDGS) and len(messages) < 20)
        calls = msg.get("tool_calls")
        if not calls:
            return (msg.get("content") or "").strip()
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            try:
                query = json.loads(call["function"]["arguments"]).get("query", "")
            except Exception:
                query = ""
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": web_search(query) if query else "No query given."})
    return (call_groq(messages, model, use_tools=False).get("content") or "").strip()


def ask_local(messages):
    r = requests.post("http://localhost:11434/api/chat",
                      json={"model": LOCAL_MODEL, "messages": messages, "stream": False}, timeout=60)
    return r.json()["message"]["content"].strip()


def ask_ai(user_text):
    print("[AI is thinking...]")
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": system_prompt()}] + history[-10:]

    reply = None
    for model in (CHAT_MODEL, BACKUP_MODEL):   # main model first, then the backup
        try:
            reply = ask_groq(messages, model)
            break
        except Exception as e:
            print(f"[{model} unavailable ({e})]")

    if reply is None:
        print("[Using local model]")
        try:
            reply = ask_local(messages)
        except Exception:
            reply = "Sorry, I can't reach my brain right now. Please check the internet."

    history.append({"role": "assistant", "content": reply})
    return reply


# =====================================================================
#                             PLUGINS
# =====================================================================

# ---------- helpers: understand durations like "5 minutes" or "twenty five minutes" ----------
ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}


def words_to_digits(text):
    text = text.replace("half an hour", "30 minutes").replace("half a minute", "30 seconds")
    for t, tv in TENS.items():                       # "twenty five" -> "25"
        for o, ov in ONES.items():
            if ov < 10:
                text = re.sub(rf"\b{t}[\s-]{o}\b", str(tv + ov), text)
    for word, value in {**TENS, **ONES}.items():
        text = re.sub(rf"\b{word}\b", str(value), text)
    text = re.sub(r"\b(a|an)\s+(hour|minute|second)\b", r"1 \2", text)  # "an hour" -> "1 hour"
    return text


def parse_seconds(text):
    """'1 hour and 30 minutes' -> 5400. Returns 0 if no duration found."""
    total = 0
    for number, unit in re.findall(r"(\d+)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
                                   words_to_digits(text)):
        n = int(number)
        total += n * 3600 if unit.startswith("h") else n * 60 if unit.startswith("m") else n
    return total


def say_duration(seconds):
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    parts = []
    if h: parts.append(f"{h} hour" + ("s" if h > 1 else ""))
    if m: parts.append(f"{m} minute" + ("s" if m > 1 else ""))
    if s: parts.append(f"{s} second" + ("s" if s > 1 else ""))
    return " and ".join(parts)


# ---------- Time and date ----------
def plugin_time(t):
    if re.search(r"\b(what time|the time|time is it|current time)\b", t):
        return "It's " + time.strftime("%I:%M %p").lstrip("0") + "."
    if re.search(r"\b(date|what day|which day)\b", t) and "update" not in t:
        return "Today is " + time.strftime("%A, %d %B %Y") + "."
    return None


# ---------- Timers and reminders ----------
def start_alarm(seconds, message):
    def ring():
        for _ in range(2):
            speak(message)
    timer = threading.Timer(seconds, ring)
    timer.daemon = True
    timer.start()
    timers.append(timer)


def plugin_timer(t):
    if "timer" in t and re.search(r"\b(cancel|stop|delete)\b", t):
        for timer in timers:
            timer.cancel()
        timers.clear()
        return "All timers and reminders are cancelled."

    if "remind me" in t:
        seconds = parse_seconds(t)
        if not seconds:
            return "When should I remind you? For example, say remind me in 10 minutes to drink water."
        task = "do your task"
        if " to " in t:
            task = t.split(" to ", 1)[1]
            task = re.sub(r"\s+in\s+(?:\S+\s+){1,3}?(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b.*$", "", task)
        start_alarm(seconds, f"Reminder: it's time to {task}.")
        return f"Okay, I'll remind you to {task} in {say_duration(seconds)}."

    if "timer" in t:
        seconds = parse_seconds(t)
        if not seconds:
            return "For how long? For example, say set a timer for 5 minutes."
        start_alarm(seconds, f"Your {say_duration(seconds)} timer is done!")
        return f"Timer set for {say_duration(seconds)}."
    return None


# ---------- Weather (Open-Meteo: free, no key needed) ----------
WEATHER_WORDS = {0: "clear skies", 1: "mostly clear skies", 2: "partly cloudy skies", 3: "cloudy skies",
                 45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                 61: "light rain", 63: "rain", 65: "heavy rain", 80: "light showers", 81: "showers",
                 82: "heavy showers", 95: "a thunderstorm", 96: "a thunderstorm with hail",
                 99: "a strong thunderstorm"}


def find_city(name):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": 1}, timeout=10)
    results = r.json().get("results")
    if not results:
        return None
    place = results[0]
    return place["name"], place["latitude"], place["longitude"]


def plugin_weather(t):
    if not re.search(r"\b(weather|rain|raining|forecast|temperature outside|hot outside|cold outside)\b", t):
        return None
    match = re.search(r"\b(?:in|at|for)\s+([a-z .]+?)(?:\s+(?:today|tomorrow|now|right now))?$", t)
    city_name = match.group(1).strip() if match else HOME_CITY
    if city_name in ("today", "tomorrow", "now"):
        city_name = HOME_CITY
    try:
        place = find_city(city_name)
        if not place:
            return f"Sorry, I couldn't find a place called {city_name}."
        name, lat, lon = place
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        }, timeout=10).json()
        day = r["daily"]
        if "tomorrow" in t:
            sky = WEATHER_WORDS.get(day["weather_code"][1], "mixed weather")
            return (f"Tomorrow in {name} expect {sky}, between {round(day['temperature_2m_min'][1])} "
                    f"and {round(day['temperature_2m_max'][1])} degrees, with a "
                    f"{day['precipitation_probability_max'][1]} percent chance of rain.")
        now = r["current"]
        sky = WEATHER_WORDS.get(now["weather_code"], "mixed weather")
        return (f"It's {round(now['temperature_2m'])} degrees in {name} with {sky}. "
                f"Today's high is {round(day['temperature_2m_max'][0])} with a "
                f"{day['precipitation_probability_max'][0]} percent chance of rain.")
    except Exception as e:
        print(f"[Weather error: {e}]")
        return "Sorry, I couldn't get the weather right now."


# ---------- Shopping / to-do list (saved in orbi_list.txt) ----------
def read_list():
    if not os.path.exists(LIST_FILE):
        return []
    with open(LIST_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def save_list(items):
    with open(LIST_FILE, "w") as f:
        f.write("\n".join(items))


def plugin_list(t):
    if "list" not in t:
        return None
    items = read_list()

    add = re.search(r"\badd (.+?) to (?:my |the )?(?:shopping |to do |to-do |todo )?list", t)
    if add:
        new = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", add.group(1)) if x.strip()]
        save_list(items + new)
        said = new[0] if len(new) == 1 else ", ".join(new[:-1]) + " and " + new[-1]
        return f"Added {said} to your list."

    remove = re.search(r"\b(?:remove|delete) (.+?) from (?:my |the )?(?:shopping |to do |to-do |todo )?list", t)
    if remove:
        target = remove.group(1).strip()
        if target in items:
            items.remove(target)
            save_list(items)
            return f"Removed {target} from your list."
        return f"I couldn't find {target} on your list."

    if re.search(r"\b(clear|empty|delete|erase)\b", t):
        save_list([])
        return "Your list is now empty."

    if re.search(r"\b(what|read|tell|show|whats|what's)\b", t):
        if not items:
            return "Your list is empty."
        return f"You have {len(items)} item" + ("s" if len(items) > 1 else "") + ": " + ", ".join(items) + "."
    return None


# ---------- Music from YouTube ----------
def stop_music():
    global music_process
    if music_process and music_process.poll() is None:
        music_process.terminate()
        music_process = None
        return True
    return False


def plugin_music(t):
    global music_process
    if t in ("stop", "stop it", "pause", "be quiet", "shut up") or \
            re.search(r"\b(stop|pause|turn off)\b.*\b(music|song|playing)\b", t):
        return "Music stopped." if stop_music() else "Nothing is playing right now."

    if not t.startswith("play "):
        return None
    if yt_dlp is None:
        return "Music needs yt-dlp. In the terminal, run pip install yt-dlp."

    query = re.sub(r"\b(on youtube|the song|song|music)\b", "", t[5:]).strip()
    if not query:
        return "What should I play?"
    speak(f"Finding {query}.")
    try:
        with yt_dlp.YoutubeDL({"format": "bestaudio", "quiet": True, "noplaylist": True}) as ydl:
            song = ydl.extract_info(f"ytsearch1:{query}", download=False)["entries"][0]
        stop_music()
        music_process = subprocess.Popen(
            ["mpv", "--no-video", "--really-quiet", f"--volume={MUSIC_VOLUME}", song["url"]],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Playing {song['title']}."
    except FileNotFoundError:
        return "Music needs the mpv player. In the terminal, run sudo apt install mpv."
    except Exception as e:
        print(f"[Music error: {e}]")
        return "Sorry, I couldn't play that song."


# ---------- Volume ----------
def set_volume(change):
    """change examples: '10%+', '10%-', '60%', 'mute', 'unmute'"""
    for cmd in (["amixer", "-q", "sset", "Master", change],
                ["amixer", "-q", "-D", "pulse", "sset", "Master", change]):
        try:
            if subprocess.run(cmd, stderr=subprocess.DEVNULL).returncode == 0:
                return True
        except FileNotFoundError:
            print("[Volume needs amixer: run 'sudo apt install alsa-utils']")
            return False
    return False


def plugin_volume(t):
    exact = re.search(r"\bvolume (?:to )?(\d{1,3})", t)
    if exact:
        level = min(int(exact.group(1)), 100)
        return f"Volume set to {level} percent." if set_volume(f"{level}%") else "Sorry, I couldn't change the volume."
    if re.search(r"\b(volume up|louder|increase (?:the )?volume|turn it up)\b", t):
        return "Turning it up." if set_volume("10%+") else "Sorry, I couldn't change the volume."
    if re.search(r"\b(volume down|quieter|softer|decrease (?:the )?volume|lower (?:the )?volume|turn it down)\b", t):
        return "Turning it down." if set_volume("10%-") else "Sorry, I couldn't change the volume."
    if re.search(r"\bunmute\b", t):
        set_volume("unmute")
        return "Sound is back on."
    if re.search(r"\bmute\b", t):
        set_volume("mute")
        print("[Muted. Type 'unmute' to turn sound back on]")
        return "Muted."
    return None


# ---------- Pi health ----------
def plugin_pi_health(t):
    if not re.search(r"\b(how hot|pi temperature|cpu temperature|your temperature|are you hot|pi health)\b", t):
        return None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read()) / 1000
        mood = "That's fine." if temp < 70 else "That's quite hot, a fan or heatsink would help."
        return f"My processor is at {temp:.0f} degrees. {mood}"
    except Exception:
        return "Sorry, I couldn't read my temperature."


# ---------- Help ----------
def plugin_help(t):
    if re.search(r"\b(what can you do|help me|what are your features|your skills)\b", t) or t == "help":
        return ("I can answer questions with live web search, tell the time and weather, set timers and reminders, "
                "keep a shopping list, play music from YouTube, and change the volume.")
    return None


# The order matters: the first plugin that understands the sentence answers it
PLUGINS = [plugin_help, plugin_music, plugin_timer, plugin_list, plugin_weather,
           plugin_volume, plugin_pi_health, plugin_time]


def run_plugins(user_text):
    t = user_text.lower().strip().rstrip(".!?")
    for plugin in PLUGINS:
        reply = plugin(t)
        if reply:
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            return reply
    return None


# =====================================================================
#                             MAIN LOOP
# =====================================================================
def handle(command):
    """Plugins first (instant), otherwise ask the AI."""
    return run_plugins(command) or ask_ai(command)


def wait_for_internet():
    for attempt in range(60):
        try:
            requests.get("https://api.groq.com", timeout=5)
            return True
        except Exception:
            if attempt == 0:
                print("[Waiting for internet...]")
            time.sleep(5)
    return False


def conversation(command):
    """Answers, then keeps listening for follow-up questions until you stop talking."""
    while command:
        print(f"You said: {command}")
        if clean(command) in END_WORDS:
            break
        speak(handle(command))
        beep()                                   # "you can keep talking"
        print(f"[Listening for a follow-up ({FOLLOW_UP_SECONDS} seconds)...]")
        pcm = mic.record(wait_seconds=FOLLOW_UP_SECONDS)
        if not pcm:
            break
        command = transcribe(pcm, "Orbi.")
        found, rest = find_wake_word(command)    # "Hey ORBI" said again is fine too
        if found and rest:
            command = rest
    done_beep()


def voice_loop():
    global mic
    mic = Microphone()
    mic.calibrate()
    speak("ORBI is ready. Say hey ORBI when you need me.")
    while True:
        print("\n[Waiting for 'Hey ORBI'...]")
        pcm = mic.record(max_seconds=8)
        seconds = len(pcm) / (RATE * 2)
        if seconds < 0.4 or seconds >= 8:        # too short to be speech, or too long to be a wake word
            continue
        text = transcribe(pcm)
        if not text:
            continue
        found, command = find_wake_word(text)
        if not found:
            print(f"[Heard (not for me): {text}]")
            continue
        print("[Wake word heard!]")
        if not command:                          # just "Hey ORBI" -> now listen for the question
            beep()
            print("[Listening...]")
            pcm = mic.record(wait_seconds=7)
            command = transcribe(pcm, "Orbi.") if pcm else ""
            if not command:
                done_beep()
                continue
        conversation(command)


def typing_loop():
    print("Type 'exit' to quit. Type 'what can you do' to hear my skills.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            return
        if not user_input:
            continue
        if user_input.lower().rstrip(".!") in ("exit", "quit", "goodbye", "bye"):
            stop_music()
            speak("Goodbye!")
            break
        speak(handle(user_input))


def main():
    print("=== ORBI Voice Assistant ===")
    print(f"[Key loaded: {GROQ_API_KEY[:6]}...{GROQ_API_KEY[-4:]}  length={len(GROQ_API_KEY)}]")
    if not GROQ_API_KEY.startswith("gsk_"):
        print("[WARNING: key should start with gsk_ - delete orbi_key.txt and run again to re-enter it]")
    if DDGS is None:
        print("[Web search is off: run 'pip install ddgs --break-system-packages' to turn it on]")
    if yt_dlp is None:
        print("[Music is off: run 'pip install yt-dlp --break-system-packages' to turn it on]")
    wait_for_internet()
    if USE_MIC:
        voice_loop()
    else:
        typing_loop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_music()
        print("\n[ORBI stopped]")



