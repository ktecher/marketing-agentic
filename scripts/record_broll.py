"""Record Instagram Reel B-roll (1080x1920, H.264) of the demo UI doing real work.

    uv run python scripts/record_broll.py <clip> [--url http://localhost:8501]

Needs the UI running. Frames come straight from Chrome's screencast at 2x (sharper than
Playwright's built-in recorder), then ffmpeg speed-ramps the slow middle: typing and the answer
play at 1x, the team's thinking is squeezed to about THINK_SECONDS.
"""

import argparse
import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "broll"
W, H, DPR = 540, 960, 2          # CSS viewport; 2x gives 1080x1920 frames
THINK_SECONDS = 10               # the thinking stretch is squeezed to this long
FPS = 30

CLIPS = {
    "01_fold8_full_briefing": {"brief": "fold 8"},
    "02_talabat_competitor": {"brief": "Research Talabat's positioning in Kuwait"},
    "03_ramadan_cafe_content": {
        "brief": "Draft 3 Instagram captions for a Kuwaiti specialty café launching its Ramadan suhoor menu",
        "follow_up": "Rewrite caption 1 for Snapchat, under 80 characters",
    },
    "04_ev_trends_kuwait": {"brief": "What are people saying about electric cars in Kuwait this month?"},
    "05_sample_data_multi_agent": {
        "brief": "Research our competitor Kiraleaf, then draft 3 Instagram posts positioning Aurelle Skin against them."
    },
    "06_settings_popup": {"settings": True},
    # 1.25x UI zoom: same 1080x1920 output, everything drawn larger, so a push-in in the edit
    # stays sharp (requested by the studio for the follow-up box in Instagram's bottom zone).
    "03b_ramadan_cafe_content_zoom125": {
        "brief": "Draft 3 Instagram captions for a Kuwaiti specialty café launching its Ramadan suhoor menu",
        "follow_up": "Rewrite caption 1 for Snapchat, under 80 characters",
        "zoom": 1.25,
    },
}


class Recorder:
    """Chrome screencast frames to disk, with wall-clock timestamps."""

    def __init__(self, page, frames: Path):
        self.frames, self.log, self.t0 = frames, [], time.time()
        frames.mkdir(parents=True)
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                               "maxWidth": W * DPR, "maxHeight": H * DPR})

    def _frame(self, ev):
        n = len(self.log)
        (self.frames / f"{n:06d}.jpg").write_bytes(base64.b64decode(ev["data"]))
        self.log.append(time.time() - self.t0)
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    def now(self) -> float:
        return time.time() - self.t0

    def stop(self):
        self.cdp.send("Page.stopScreencast")


def scroll_to(page, selector: str, offset: int = 70):
    page.evaluate(
        """([sel, off]) => { const el = document.querySelector(sel); if (!el) return;
             const y = el.getBoundingClientRect().top + window.scrollY - off;
             const main = document.querySelector('section.stMain, [data-testid="stMain"]') || window;
             (main.scrollTo ? main : window).scrollTo({top: y, behavior: 'smooth'}); }""",
        [selector, offset],
    )


def smooth_scroll(page, total_px: int, seconds: float):
    page.mouse.move(W / 2, H / 2)  # the wheel scrolls whatever is under the pointer
    steps = int(seconds * 30)
    for _ in range(steps):
        page.mouse.wheel(0, total_px / steps)
        page.wait_for_timeout(1000 / 30)


def wait_done(page, timeout_s: int = 600):
    """A run shows 'Working for …' in its status line and 'Finished in …' when done."""
    page.locator("text=Working for").first.wait_for(timeout=60000)
    deadline = time.time() + timeout_s
    while page.locator("text=Working for").count():
        if time.time() > deadline:
            raise TimeoutError("run did not finish")
        page.wait_for_timeout(1000)


def run_brief(page, rec, brief: str, marks: dict, key: str):
    box = page.get_by_placeholder("For example", exact=False)
    box.click()
    marks[f"{key}_type"] = rec.now()
    page.keyboard.type(brief, delay=45)
    page.wait_for_timeout(600)
    page.get_by_role("button", name="Brief the team").click()
    marks[f"{key}_submit"] = rec.now()
    page.wait_for_timeout(1500)
    scroll_to(page, ".cell-stage")
    wait_done(page)
    marks[f"{key}_final"] = rec.now()


def record(clip: str, url: str):
    global W, H, DPR
    spec = CLIPS[clip]
    zoom = spec.get("zoom", 1.0)
    # Same 1080x1920 output: shrink the CSS viewport and raise the pixel ratio by the zoom.
    W, H, DPR = round(540 / zoom), round(960 / zoom), 2 * zoom
    work = OUT / ".work" / clip
    shutil.rmtree(work, ignore_errors=True)
    marks: dict[str, float] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=DPR)
        page = ctx.new_page()
        page.goto(url)
        page.get_by_role("button", name="Brief the team").wait_for(timeout=60000)
        page.wait_for_timeout(2500)
        rec = Recorder(page, work / "frames")
        page.wait_for_timeout(2500)

        if spec.get("settings"):
            # Phone width folds the sidebar away: open it, then the Settings pop-up.
            page.get_by_test_id("stExpandSidebarButton").click()
            page.wait_for_timeout(1500)
            page.get_by_role("button", name="Settings").click()
            page.get_by_text("Search sources").wait_for(timeout=15000)
            page.wait_for_timeout(3000)
            smooth_scroll(page, 900, 4)
            page.wait_for_timeout(2500)
            marks["end"] = rec.now()
        else:
            run_brief(page, rec, spec["brief"], marks, "a")
            page.wait_for_timeout(3500)                 # the nucleus flare, then the answer below
            scroll_to(page, "h3, [data-testid='stHeading']", 40)
            page.wait_for_timeout(1200)
            smooth_scroll(page, 1400, 7)
            page.wait_for_timeout(1500)
            if follow := spec.get("follow_up"):
                chat = page.get_by_placeholder("Ask a follow-up", exact=False)
                chat.click()
                marks["b_type"] = rec.now()
                page.keyboard.type(follow, delay=45)
                page.wait_for_timeout(500)
                page.keyboard.press("Enter")
                marks["b_submit"] = rec.now()
                scroll_to(page, ".cell-stage", 70)
                wait_done(page)
                marks["b_final"] = rec.now()
                page.wait_for_timeout(2000)
                smooth_scroll(page, 1200, 6)
                page.wait_for_timeout(1500)
            marks["end"] = rec.now()
        rec.stop()
        page.wait_for_timeout(300)
        browser.close()
    (work / "marks.json").write_text(json.dumps({"marks": marks, "frames": rec.log}, indent=1))
    encode(clip, work, rec.log, marks)


def encode(clip: str, work: Path, times: list[float], marks: dict):
    """Speed-ramp: 1x everywhere except submit+2s .. final-1s, squeezed to THINK_SECONDS."""
    slow = []  # (start, end) stretches to squeeze
    for k in ("a", "b"):
        if f"{k}_submit" in marks:
            s, e = marks[f"{k}_submit"] + 2, marks[f"{k}_final"] - 1
            if e - s > THINK_SECONDS:
                slow.append((s, e))

    def out_time(t: float) -> float:
        shift = 0.0
        for s, e in slow:
            if t <= s:
                break
            factor = (e - s) / THINK_SECONDS
            if t < e:
                return t - shift - (t - s) * (1 - 1 / factor)
            shift += (e - s) * (1 - 1 / factor)
        return t - shift

    # One source frame per output frame at FPS: the latest frame shown by then. (Passing every
    # frame with tiny durations does not work: the demuxer rounds each one up to a whole tick,
    # so squeezed stretches came out 3x too long.)
    import bisect
    out_t = [out_time(t) for t in times]
    lines = []
    for k in range(int(out_t[-1] * FPS) + 1):
        i = max(0, bisect.bisect_right(out_t, k / FPS) - 1)
        lines += [f"file 'frames/{i:06d}.jpg'", f"duration {1 / FPS:.6f}"]
    lines.append(lines[-2])
    (work / "list.txt").write_text("\n".join(lines))
    OUT.mkdir(exist_ok=True)
    dest = OUT / f"{clip}.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
        # JPEG frames are full range; video players expect TV range.
        "-vf", f"scale={W * DPR}:{H * DPR}:flags=lanczos:in_range=pc:out_range=tv,fps={FPS},format=yuv420p",
        "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-movflags", "+faststart",
        str(dest),
    ], check=True)
    print(f"{dest}  ({out_t[-1]:.1f}s, from {times[-1]:.0f}s recorded, {len(times)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", choices=sorted(CLIPS))
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--encode-only", action="store_true", help="re-encode from the frames already recorded")
    a = ap.parse_args()
    if a.encode_only:
        work = OUT / ".work" / a.clip
        saved = json.loads((work / "marks.json").read_text())
        encode(a.clip, work, saved["frames"], saved["marks"])
    else:
        record(a.clip, a.url)
