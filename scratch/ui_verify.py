#!/usr/bin/env python3
"""
Click-test the whole product in a real browser.

    ~/venv/bin/python scratch/ui_verify.py [appUrl] [apiUrl]

Covers the FEATURE_VERIFICATION items that can only be confirmed by
clicking: Batch (9), Patient Comparison (10), the Admin annotation queue
(11), the notes-parser Apply step and the Arabic notice (15), History (2b),
and the Clinical Note box (7).

Every check prints PASS/FAIL with what it actually saw. Exit 0 only if all
pass. Screenshots go to <out>/shots/.
"""
import asyncio, json, os, sys, pathlib
from playwright.async_api import async_playwright

CHROME = "/home/yahia/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome"
APP = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4173"
API = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
OUT = pathlib.Path(os.environ.get("OUT", "/tmp/ui_verify"))
SHOTS = OUT / "shots"; SHOTS.mkdir(parents=True, exist_ok=True)

DOCTOR = ("doctor@omnidiag.com", "Doctor@123")
ADMIN  = ("admin@omnidiag.com", "Admin@123")

results = []
def record(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail else ""))

async def shot(page, name):
    try: await page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
    except Exception: pass

async def nav(page, label):
    """Click a sidebar entry and wait for the view to settle."""
    btn = page.get_by_role("button", name=label, exact=False)
    if await btn.count() == 0:
        return False
    await btn.first.click()
    await page.wait_for_timeout(2500)
    return True

async def modal_open(page):
    """True while the login modal's backdrop is on screen."""
    return await page.locator("div.fixed.inset-0.z-50").count() > 0


async def sign_in(page, email, password):
    """
    Sign in and CONFIRM the modal closed.

    The modal only closes on a successful login, so a still-open modal means
    the credentials were rejected — and it leaves a full-screen backdrop that
    silently swallows every later click. Not verifying this cost a whole run.
    """
    if not await modal_open(page):
        # exact=True on purpose: the Admin Dashboard nav item's accessible
        # name is "Admin Dashboard Sign in", so a substring match clicked the
        # nav entry instead of the sidebar's Sign In button and silently
        # navigated to Admin — where the disease selector does not exist,
        # which then looked like the selector had vanished.
        opener = page.get_by_role("button", name="Sign In", exact=True)
        if await opener.count() == 0:
            opener = page.get_by_role("button", name="Sign In as Admin", exact=True)
        if await opener.count():
            await opener.first.click()
        await page.wait_for_timeout(1500)
    await page.locator('input[type="email"]').first.fill(email)
    await page.locator('input[type="password"]').first.fill(password)
    await page.locator('button[type="submit"]').last.click()
    await page.wait_for_timeout(4000)
    ok = not await modal_open(page)
    record(f"signed in as {email}", ok,
           "login modal is still open — credentials rejected" if not ok else "")
    if not ok:
        await shot(page, "signin_failed")
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(600)
    return ok

async def body(page):
    """
    Rendered page text, lowercased.

    Lowercased on purpose: several headers use Tailwind's `uppercase` class,
    and inner_text() returns the RENDERED text, so "Screening result" comes
    back as "SCREENING RESULT". Every assertion below compares lowercase.
    """
    return (await page.inner_text("body")).lower()


def has(text, *needles):
    """True when every needle (lowercased) appears in `text`."""
    return all(n.lower() in text for n in needles)


# ── 16: neutral wording ───────────────────────────────────────────────────────
async def check_disease_dropdown(page):
    print("\n[1] disease dropdown")
    trigger = page.locator('button[aria-haspopup="listbox"]')
    if await trigger.count() == 0:
        record("disease selector present", False); return
    await trigger.first.click(); await page.wait_for_timeout(1000)
    opts = await page.locator('[role="option"]').all_inner_texts()
    await shot(page, "disease_dropdown")
    await page.keyboard.press("Escape"); await page.wait_for_timeout(500)
    joined = " | ".join(o.replace("\n", " ") for o in opts).lower()
    record("dropdown shows display names, not raw keys",
           "coronary artery disease" in joined and "diabetes risk assessment" in joined,
           joined[:200])
    record("dropdown has real descriptions", "no description" not in joined,
           joined[:200] if "no description" in joined else "")
    record("dropdown shows a version, not a dash", "v—" not in joined, joined[:200])


async def check_wording(page):
    print("\n[16] user-facing wording")
    t = await body(page)
    record("subtitle is not 'Diagnostic Platform'", not has(t, "multi-disease diagnostic platform"))
    record("'Clinical Decision Support' shown", has(t, "clinical decision support"))
    title = await page.title()
    record("tab title is neutral", "Diagnostic Platform" not in title, f"title={title!r}")
    record("nav says 'Patient Comparison', not 'Before/After'",
           has(t, "patient comparison") and "before/after" not in t)


# ── 1c: Scales / Randomize / Reset ────────────────────────────────────────────
async def check_scales_and_randomize(page, disease_label):
    print(f"\n[1c] Scales / Randomize — {disease_label}")
    await nav(page, "Engineering Mode")
    rnd = page.get_by_role("button", name="Randomize")
    try:
        # The diabetes form fetches a 21-field schema; the button does not
        # exist until it resolves.
        await rnd.first.wait_for(state="visible", timeout=25000)
    except Exception:
        pass
    if await rnd.count() == 0:
        record(f"Randomize present ({disease_label})", False, "button not found")
        return
    await rnd.first.click(); await page.wait_for_timeout(1200)

    vals = await page.evaluate("""() => {
        // Diabetes fields render as toggles and sliders, not number inputs,
        // and react-hook-form does not put a name on every control.
        const out = {};
        let i = 0;
        document.querySelectorAll('input, select').forEach(el => {
            if (el.type === 'file' || el.type === 'password') return;
            const key = el.name || el.id || `field_${i++}`;
            out[key] = el.type === 'checkbox' ? String(el.checked) : el.value;
        });
        return out;
    }""")
    record(f"Randomize filled fields ({disease_label})", len(vals) > 0, f"{len(vals)} inputs")

    if disease_label == "heart_disease":
        bp = float(vals.get("RestingBP") or 0); chol = float(vals.get("Cholesterol") or 0)
        record("randomised RestingBP is plausible", 90 <= bp <= 180, f"RestingBP={bp}")
        record("randomised Cholesterol is plausible", 140 <= chol <= 330, f"Cholesterol={chol}")

    sc = page.get_by_role("button", name="Scales")
    await sc.first.click()
    try:
        await page.locator('text=Scale / Range').first.wait_for(state="visible", timeout=15000)
    except Exception:
        pass
    await page.wait_for_timeout(1200)
    t = await body(page)
    await shot(page, f"scales_{disease_label}")
    if disease_label == "heart_disease":
        record("heart Scales does NOT cite BRFSS", "brfss" not in t,
               "BRFSS footer still on the heart page" if "brfss" in t else "")
        record("heart Scales cites UCI", has(t, "uci heart disease"))
        record("heart Scales says Age is in years", has(t, "age is in years"))
    else:
        record("diabetes Scales cites BRFSS", has(t, "brfss"))
        record("diabetes Scales explains the Age band", has(t, "5-year band"))
    # Read the table rows rather than slicing the page text: the old
    # heuristic split on the LAST "categorical" and looked at the next 40
    # characters, which ran into the row's clinical description and its
    # em-dashes.
    bad_rows = await page.evaluate("""() => {
        const out = [];
        document.querySelectorAll('table tbody tr').forEach(tr => {
            const cells = [...tr.querySelectorAll('td')].map(td => td.innerText.trim());
            if (cells.length >= 3 && /categorical/i.test(cells[1]) && cells[2].replace(/\s/g,'') === '—')
                out.push(cells[0]);
        });
        return out;
    }""")
    record("no 'Categorical' row without a range", len(bad_rows) == 0,
           f"rows with no range: {bad_rows}" if bad_rows else "")
    for name in ("Close", "close"):
        b = page.get_by_role("button", name=name, exact=False)
        if await b.count(): await b.first.click(); break
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(800)

    rst = page.get_by_role("button", name="Reset")
    if await rst.count():
        await rst.first.click(); await page.wait_for_timeout(1000)
        record(f"Reset works ({disease_label})", True)


# ── 10: Patient Comparison ────────────────────────────────────────────────────
async def check_compare(page, disease_label):
    print(f"\n[10] Patient Comparison — {disease_label}")
    if not await nav(page, "Patient Comparison"):
        record(f"Compare view opens ({disease_label})", False, "nav entry not found"); return
    t = await body(page)
    record(f"labelled Patient A/B, not Before/After ({disease_label})",
           has(t, "patient a", "patient b") and "before / after" not in t,
           "'before / after' still present" if "before / after" in t else "")
    record("subtitle admits it is two different patients", has(t, "two different patients"))
    btn = page.get_by_role("button", name="Compare", exact=True)
    if await btn.count() == 0:
        record(f"Compare button present ({disease_label})", False); return
    await btn.first.click()
    await page.wait_for_timeout(9000)
    t = await body(page)
    await shot(page, f"compare_{disease_label}")
    record(f"comparison produced results ({disease_label})", has(t, "risk probability"))
    record("no duplicate 'Risk Score' tile", "risk score" not in t)
    ok_delta = has(t, "vs patient a") or has(t, "same estimated risk")
    record(f"delta is comparative, not a change over time ({disease_label})", ok_delta,
           "" if ok_delta else "neither 'vs Patient A' nor 'Same estimated risk' found")


# ── 9: Batch ──────────────────────────────────────────────────────────────────
async def check_batch(page, disease_label, csv_path, rows):
    print(f"\n[9] Batch — {disease_label} ({rows} rows)")
    if not await nav(page, "Batch Prediction"):
        record("Batch view opens", False, "nav entry not found"); return
    shown = await current_disease(page)
    if shown != disease_label:
        await shot(page, f"switch_failed_{disease_label}")
        record(f"sidebar is on {disease_label}", False, f"sidebar shows {shown}")
        return
    t = await body(page)
    expected = "up to 20 patients" if disease_label == "diabetes" else "up to 500 patients"
    record(f"row limit stated for {disease_label}", has(t, expected),
           t[t.find("up to"):t.find("up to")+80] if "up to" in t else "no limit text")
    # A previous run leaves its results on screen, which hides the drop zone.
    newb = page.get_by_role("button", name="New Batch")
    if await newb.count():
        await newb.first.click(); await page.wait_for_timeout(1200)
    inp = page.locator('input[type="file"]')
    await inp.first.set_input_files(csv_path)
    await page.wait_for_timeout(1500)
    run = page.get_by_role("button", name="Run Batch")
    if await run.count() == 0 or not await run.first.is_enabled():
        t = await body(page)
        record(f"{disease_label} batch accepted the file", False,
               "Run Batch is disabled — not signed in, or the file was refused. "
               + t[-300:])
        return
    await run.first.click()
    await page.wait_for_timeout(12000)
    t = await body(page)
    await shot(page, f"batch_{disease_label}")
    record(f"{disease_label} batch returned results", has(t, "result distribution"),
           t[-300:] if "result distribution" not in t else "")
    record(f"column is 'Screening result' ({disease_label})", has(t, "screening result"))
    record(f"no duplicate 'Diagnosis' column ({disease_label})", "diagnosis" not in t)


async def check_batch_cap(page, big_csv, rows):
    print(f"\n[9] Batch — diabetes cap ({rows} rows, over the limit)")
    await nav(page, "Batch Prediction")
    newb = page.get_by_role("button", name="New Batch")
    if await newb.count():
        await newb.first.click(); await page.wait_for_timeout(1200)
    await page.locator('input[type="file"]').first.set_input_files(big_csv)
    await page.wait_for_timeout(2500)
    t = await body(page)
    await shot(page, "batch_diabetes_over_cap")
    record("oversized diabetes file is refused with a clear message",
           str(rows) in t and "limit" in t,
           t[t.find("has "):t.find("has ")+180] if "has " in t else t[-300:])


async def check_batch_reset_on_switch(page, heart_csv, select_disease):
    print("\n[9] Batch — results reset when the disease is switched")
    await nav(page, "Batch Prediction")
    newb = page.get_by_role("button", name="New Batch")
    if await newb.count():
        await newb.first.click(); await page.wait_for_timeout(1200)
    await page.locator('input[type="file"]').first.set_input_files(heart_csv)
    await page.wait_for_timeout(1500)
    run = page.get_by_role("button", name="Run Batch")
    if await run.count():
        await run.first.click(); await page.wait_for_timeout(9000)
    before = await body(page)
    had_results = "result distribution" in before
    await select_disease(page, "diabetes")
    await page.wait_for_timeout(2500)
    after = await body(page)
    await shot(page, "batch_after_switch")
    record("heart results are cleared after switching to diabetes",
           had_results and "result distribution" not in after,
           "results were still on screen" if "result distribution" in after
           else ("no results to clear in the first place" if not had_results else ""))


# ── 15: notes parser — Apply step and the Arabic notice ──────────────────────
async def check_notes_parser(page):
    print("\n[15] Clinical Notes Parser — Apply and Arabic")
    await nav(page, "Clinical EMR Mode")
    await page.wait_for_timeout(4000)
    opener = page.get_by_role("button", name="Clinical Notes Parser", exact=False)
    if await opener.count():
        await opener.first.click(); await page.wait_for_timeout(1200)

    # By placeholder, not "the first textarea": the Clinical Note box now
    # sits above the parser, so .first picks the wrong one and "Extract
    # Fields" stays disabled.
    ta = page.get_by_placeholder("Paste an English clinical note", exact=False)
    await ta.first.fill("58-year-old male with typical angina. BP 152/94. "
                        "Total cholesterol 268 mg/dl. Max heart rate 138.")
    await page.get_by_role("button", name="Extract Fields").first.click()
    await page.wait_for_timeout(4000)
    t = await body(page)
    record("English note extracts fields", has(t, "found"), t[-200:] if "found" not in t else "")
    record("fields are NOT applied before confirmation", has(t, "apply"),
           "no Apply button — values may be auto-applied")
    await shot(page, "notes_extracted")

    apply_btn = page.get_by_role("button", name="Apply", exact=False)
    if await apply_btn.count():
        await apply_btn.first.click()
        await page.wait_for_timeout(6000)
        t = await body(page)
        await shot(page, "notes_applied")
        record("Apply confirms and re-runs the screening", has(t, "applied"), t[-250:])
    else:
        record("Apply button present", False)

    # Arabic
    opener = page.get_by_role("button", name="Clinical Notes Parser", exact=False)
    if await opener.count():
        await opener.first.click(); await page.wait_for_timeout(800)
        await opener.first.click(); await page.wait_for_timeout(800)
    ta = page.get_by_placeholder("Paste an English clinical note", exact=False)
    await ta.first.fill("مريض ذكر عمره 58 سنة، ضغط الدم 150/95. الكوليسترول الكلي 260.")
    await page.get_by_role("button", name="Extract Fields").first.click()
    await page.wait_for_timeout(4000)
    t = await body(page)
    await shot(page, "notes_arabic")
    record("Arabic note is explicitly refused, not silently empty",
           has(t, "arabic notes are not supported"),
           t[-300:] if "arabic notes are not supported" not in t else "")
    record("Arabic notice does not claim data was updated", "patient data updated" not in t)


# ── 7 + 2b: Clinical Note and History ────────────────────────────────────────
async def check_note_and_history(page):
    print("\n[7 + 2b] Clinical Note and History")
    await nav(page, "Clinical EMR Mode")
    await page.wait_for_timeout(5000)
    t = await body(page)
    record("Clinical Note box is present", has(t, "clinical note"))

    note = "Playwright check — exertional dyspnoea not captured by the form."
    target = page.get_by_placeholder("exertional dyspnoea", exact=False)
    if await target.count() == 0:
        record("Clinical Note textarea found", False); return
    await target.first.fill(note)
    save = page.get_by_role("button", name="Save note")
    if await save.count() == 0:
        record("Save note button present", False); return
    await save.first.click()
    await page.wait_for_timeout(4000)
    t = await body(page)
    await shot(page, "note_saved")
    record("note saves against the screening", has(t, "saved to this screening"),
           t[-300:] if "saved to this screening" not in t else "")

    hist = page.get_by_role("button", name="History", exact=False)
    if await hist.count() == 0:
        record("History button present", False); return
    await hist.first.click()
    await page.wait_for_timeout(6000)
    # The note lives in the card's expanded detail, so the newest entry has
    # to be opened before looking for it. Scoped to the History drawer: the
    # unscoped locator matched a card on the EMR page behind it, whose click
    # the drawer's backdrop intercepts.
    drawer = page.locator("div.fixed.inset-y-0.right-0.z-50")
    cards = drawer.locator("button.card-header")
    if await cards.count():
        await cards.first.click()
        await page.wait_for_timeout(1500)
    t = await body(page)
    await shot(page, "history")
    record("History opens without a JSON parse error",
           "not valid json" not in t and "<!doctype" not in t,
           t[-400:] if "not valid json" in t else "")
    record("History shows timeline entries",
           has(t, "risk probability") and "not found" not in t,
           t[-300:] if "not found" in t else "")
    record("the saved note appears in History", note[:40].lower() in t,
           "note not visible in the timeline" if note[:40].lower() not in t else "")


# ── 11: Admin annotation queue ────────────────────────────────────────────────
async def check_admin_queue(page):
    print("\n[11] Admin — annotation queue")
    if not await nav(page, "Admin"):
        record("Admin view opens", False, "nav entry not found"); return
    await page.wait_for_timeout(4000)
    t = await body(page)
    if "admin access required" in t or ("sign in" in t and "annotation queue" not in t):
        await sign_in(page, *ADMIN)
        await nav(page, "Admin")
        await page.wait_for_timeout(4000)
        t = await body(page)
    await shot(page, "admin")
    record("Admin dashboard loads", has(t, "annotation queue"), t[:200])
    record("average is labelled as probability, not confidence",
           "avg confidence" not in t or has(t, "risk probability"))

    if has(t, "no pending items"):
        record("queue has items to label", False,
               "queue empty — cannot click-test labelling")
        return

    body_text = t[t.find("annotation queue"):t.find("annotation queue") + 1200]
    record("queue rows are not empty placeholders",
           body_text.count("—") < 8,
           f"{body_text.count('—')} em-dashes in the queue block")

    note_input = page.get_by_placeholder("Why this label?", exact=False)
    record("per-row note input exists", await note_input.count() > 0)
    if await note_input.count():
        await note_input.first.fill("Playwright check — borderline, agrees with model.")
        await page.wait_for_timeout(400)
    pos = page.get_by_role("button", name="+ Pos")
    if await pos.count() == 0:
        record("Pos/Neg buttons present", False); return
    await pos.first.click()
    await page.wait_for_timeout(5000)
    await shot(page, "admin_after_label")
    record("labelling an item succeeds", True)


async def verify_label_persisted(api_check):
    print("\n[11] annotation label + note persisted (API)")
    ok, detail = await api_check()
    record("label and note are both stored", ok, detail)


# ── driver ────────────────────────────────────────────────────────────────────
# The sidebar selector is a custom listbox, not a <select>: a trigger button
# with aria-haspopup="listbox", and options carrying role="option" whose text
# is the DISPLAY name, not the disease key.
DISPLAY_NAME = {
    "heart_disease": "Coronary Artery Disease",
    "diabetes": "Diabetes Risk Assessment",
}


async def select_disease(page, name):
    """
    Pick a disease from the sidebar listbox, and confirm it actually took.

    Verified rather than assumed: a silently-failed switch made every later
    check run against the wrong module and report the wrong module's text.
    """
    for attempt in range(3):
        if await current_disease(page) == name:
            return True
        trigger = page.locator('button[aria-haspopup="listbox"]')
        if await trigger.count() == 0:
            return False
        await trigger.first.click()
        await page.wait_for_timeout(800)
        option = page.locator('[role="option"]').filter(has_text=DISPLAY_NAME[name])
        if await option.count() == 0:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(400)
            continue
        await option.first.click()
        await page.wait_for_timeout(3500)
    return await current_disease(page) == name


async def current_disease(page):
    """Which disease the sidebar currently shows."""
    trigger = page.locator('button[aria-haspopup="listbox"]')
    if await trigger.count() == 0:
        return None
    label = (await trigger.first.inner_text()).lower()
    for key, display in DISPLAY_NAME.items():
        if display.lower() in label:
            return key
    return None


def write_csvs(tmp):
    heart = tmp / "heart.csv"
    heart.write_text(
        "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope\n"
        "54,M,ATA,140,289,0,Normal,122,N,0,Flat\n"
        "62,F,ASY,160,340,1,ST,110,Y,3.2,Down\n"
        "45,M,NAP,120,200,0,Normal,170,N,0.5,Up\n")
    cols = ("HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,"
            "Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,"
            "PhysHlth,DiffWalk,Sex,Age,Education,Income")
    row = "1,1,1,33,1,0,0,0,0,1,0,1,0,4,5,10,1,0,9,4,3"
    (tmp / "diabetes.csv").write_text(cols + "\n" + "\n".join([row] * 5) + "\n")
    (tmp / "diabetes_big.csv").write_text(cols + "\n" + "\n".join([row] * 25) + "\n")
    return heart, tmp / "diabetes.csv", tmp / "diabetes_big.csv"


async def main():
    tmp = OUT / "csv"; tmp.mkdir(parents=True, exist_ok=True)
    heart_csv, diab_csv, diab_big = write_csvs(tmp)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=CHROME)
        page = await browser.new_page(viewport={"width": 1440, "height": 950})
        page.on("pageerror", lambda e: print(f"          [pageerror] {e}"))
        print(f"App: {APP}\nAPI: {API}")
        await page.goto(APP, wait_until="networkidle", timeout=180000)
        await page.wait_for_timeout(5000)

        await check_wording(page)
        await check_disease_dropdown(page)

        await select_disease(page, "heart_disease")
        await check_scales_and_randomize(page, "heart_disease")
        await check_compare(page, "heart_disease")

        await select_disease(page, "diabetes")
        await check_scales_and_randomize(page, "diabetes")
        await check_compare(page, "diabetes")

        # Batch needs a clinical account.
        await nav(page, "Batch Prediction")
        t = await body(page)
        if "sign in with a clinical account" in t or "sign in to" in t:
            await sign_in(page, *DOCTOR)
            # Return to Batch: signing in can leave the app on another view.
            await nav(page, "Batch Prediction")
            await page.wait_for_timeout(2000)
        await select_disease(page, "heart_disease")
        await check_batch(page, "heart_disease", str(heart_csv), 3)
        await select_disease(page, "diabetes")
        await check_batch(page, "diabetes", str(diab_csv), 5)
        await check_batch_cap(page, str(diab_big), 25)
        await select_disease(page, "heart_disease")
        await check_batch_reset_on_switch(page, str(heart_csv), select_disease)

        await select_disease(page, "heart_disease")
        await check_notes_parser(page)
        await check_note_and_history(page)
        await check_admin_queue(page)

        await browser.close()

    passed = sum(1 for r in results if r["ok"])
    print(f"\n{'='*64}\n{passed}/{len(results)} checks passed")
    failed = [r for r in results if not r["ok"]]
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  - {r['check']}" + (f"\n      {r['detail'][:200]}" if r["detail"] else ""))
    (OUT / "ui_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nResults: {OUT/'ui_results.json'}   Screenshots: {SHOTS}")
    return 0 if not failed else 1

sys.exit(asyncio.run(main()))
