import random
import csv
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
NUM_SAMPLES_PER_CATEGORY = 350
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "alerts_dataset_v2.csv")

# Banks and common data
BANKS = [
    "GTBank",
    "Access Bank",
    "Zenith Bank",
    "UBA",
    "FirstBank",
    "Stanbic IBTC",
    "Fidelity Bank",
    "Opay",
    "Kuda",
    "Moniepoint",
]
ACCOUNT_TYPES = ["Savings", "Current"]
CURRENCIES = ["NGN", "USD"]

# Templates
LEGITIMATE_TEMPLATES = [
    "Acct: {account} {type}: {currency}{amount} Date: {date} Bal: {currency}{balance} {narration}",
    "Transaction Alert: {type} {currency}{amount} on {account} Date: {date} Nar: {narration} Bal: {currency}{balance}",
    "{bank} ALERT: {type} {currency}{amount} to {recipient} Ref: {ref} {date} Bal: {currency}{balance}",
    "Zenith Bank Alert [CREDIT: {currency}{amount}] {date} {account} {narration} Bal:{currency}{balance}",
    "UBA Credit Alert: Acct: {account} Amt: {currency}{amount} Date: {date} Bal: {currency}{balance} Descr: {narration}",
    "Notification: Your {account} was credited with {currency}{amount} from {recipient}. Bal: {currency}{balance}",
    "Debit: {currency}{amount} | {account} | {date} | {narration} | Bal: {currency}{balance}",
    "{bank} Alert: {type} {currency}{amount} {narration} Bal {currency}{balance}",
    "{bank} {type} {currency}{amount} {date} Ref {ref}. Bal {currency}{balance}",
]

FAKE_TEMPLATES = [
    # Explicit Phishing
    "Urgent: Your {bank} account has been suspended due to BVN irregularity. Click here to verify now: {link}",
    "Dear Customer, your BVN will be deactivated in 24 hours. Follow this link to update your details: {link}",
    "Suspicious activity detected on your {bank} account. Your ATM has been blocked. Call {phone} to reactivate.",
    "You have a pending refund of {currency}{amount} from {bank}. Provide your card details here to claim: {link}",
    "Alert: Your BVN validation failed. To avoid total account restriction, update at: {link}",
    "Dear customer, you won 50,000 NGN in the {bank} promo. Send your OTP to {phone} to receive funds.",
    # Subtle Phishing / Misspellings
    "URGENT: Your {bank} acount has bin restricted due to central bank regulatons. Pls login here to verify: {link}",
    "Cbn News: All BVN customers are requierd to link there NIN to avoid account closure. Click {link} to avoid blockage.",
    "Dear {bank} User, suspicious transacion of {currency}{amount} was noticed. If not you, cancel it here: {link}",
    "Alert: You recieved a reward of {currency}{amount} from GTB. Claim within 2hrs: {link}",
    "G-T-B-A-N-K Notification: Your mobile app session is expiring. Login to stay active: {link}",
    "Your {bank} account is on hold. Verify now at {link} to avoid restriction.",
    "Action required: confirm your BVN to keep account active. Link: {link}",
]

SUSPICIOUS_TEMPLATES = [
    "Notification: Login at {time} on {device} from {location}. Not you? Click {link}",
    "Account {account} was accessed from a new device. If this was not you, please contact support or click {link}",
    "Password change request for your {bank} mobile app received. Use code {otp} to continue.",
    "Your {bank} account profile was modified on {date}. If you did not make this change, visit {link}",
    "Security Alert: Your {bank} internet banking password was changed from {location}. Contact us if not you.",
    "Alert: A new device {device} has been linked to your {bank} profile. Not you? Secure account at {link}",
    "{bank} Notice: unusual login attempt detected. If not you, reset now: {link}",
    "Security notice: new beneficiary added on your {bank} account. Ignore if this was you.",
]


def generate_random_data():
    account = f"***{random.randint(100, 999)}4{random.randint(100, 999)}"
    amount = f"{random.randint(100, 500000):,.2f}"
    balance = f"{random.randint(1000, 2000000):,.2f}"
    date = f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-2026"
    time = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
    narration = random.choice(
        [
            "ATM WDL",
            "POS/PURCHASE",
            "WEB/IB-TRF",
            "TRF-FROM",
            "NIP-TRANSFER",
            "USSD Transfer",
            "Airtime Topup",
        ]
    )

    # Fake links with common typos or lookalikes
    link = random.choice(
        [
            "bit.ly/bank-verify",
            "tinyurl.com/bvn-update",
            "gtbank-online.xyz",
            "access-secure.net",
            "u-b-a-portal.com",
            "v-erify-bvn.ng",
            "zenith-bank-login.com",
            "mybnk-update.top",
        ]
    )

    phone = f"080{random.randint(10000000, 99999999)}"
    device = random.choice(
        [
            "iPhone 13",
            "Samsung S21",
            "MacBook Pro",
            "Windows Desktop",
            "Infinix Hot 10",
            "Tecno Spark",
        ]
    )
    location = random.choice(
        ["Lagos", "Abuja", "Port Harcourt", "London", "USA", "Dubai", "Lekki"]
    )
    ref = f"{random.randint(1000000, 9999999)}"
    otp = f"{random.randint(100000, 999999)}"

    return {
        "bank": random.choice(BANKS),
        "account": account,
        "type": random.choice(["CREDIT", "DEBIT"]),
        "currency": random.choice(CURRENCIES),
        "amount": amount,
        "balance": balance,
        "date": date,
        "time": time,
        "narration": narration,
        "link": link,
        "phone": phone,
        "device": device,
        "location": location,
        "ref": ref,
        "otp": otp,
        "recipient": random.choice(
            [
                "OLAYINKA BAKARE",
                "CHUKWUMA OKORO",
                "MUSA BELLO",
                "AMINA YUSUF",
                "KELECHI NWOSU",
            ]
        ),
        "support": random.choice(
            [
                "customer care",
                "support",
                "helpdesk",
                "contact center",
            ]
        ),
    }


def _random_case(text: str) -> str:
    if random.random() < 0.3:
        return text.upper()
    if random.random() < 0.3:
        return text.lower()
    return "".join(
        ch.upper() if random.random() < 0.08 else ch for ch in text
    )


def _inject_punctuation(text: str) -> str:
    if random.random() < 0.4:
        return text.replace(" ", random.choice(["  ", " ", " | ", " - "]))
    return text


def _introduce_typos(text: str) -> str:
    replacements = {
        "account": "acount",
        "verify": "verfy",
        "urgent": "urgnt",
        "login": "logn",
        "security": "securrity",
        "notification": "notificaton",
        "received": "recieved",
        "transaction": "transacion",
    }
    for src, tgt in replacements.items():
        if src in text.lower() and random.random() < 0.2:
            text = text.replace(src, tgt).replace(src.capitalize(), tgt.capitalize())
    return text


def _add_noise(text: str) -> str:
    text = _random_case(text)
    text = _inject_punctuation(text)
    text = _introduce_typos(text)
    if random.random() < 0.25:
        text = f"{text} {random.choice(['pls', 'kindly', 'now', 'asap'])}"
    return text


def _mix_signal(text: str) -> str:
    # Add borderline cues to reduce trivial separability
    if random.random() < 0.2:
        text = f"{text} If not you, call {random.choice(['customer care', 'support'])}."
    if random.random() < 0.15:
        text = f"{text} Do not share your OTP."
    return text


def create_dataset():
    data = []

    # Generate Legitimate (0)
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        params = generate_random_data()
        template = random.choice(LEGITIMATE_TEMPLATES)
        msg = template.format(**params)
        msg = _add_noise(_mix_signal(msg))
        data.append({"text": msg, "label": 0})

    # Generate Fake (1)
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        params = generate_random_data()
        template = random.choice(FAKE_TEMPLATES)
        msg = template.format(**params)
        msg = _add_noise(_mix_signal(msg))
        data.append({"text": msg, "label": 1})

    # Generate Suspicious (2)
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        params = generate_random_data()
        template = random.choice(SUSPICIOUS_TEMPLATES)
        msg = template.format(**params)
        msg = _add_noise(_mix_signal(msg))
        data.append({"text": msg, "label": 2})

    random.shuffle(data)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Generated {len(data)} samples in {OUTPUT_FILE}")


if __name__ == "__main__":
    create_dataset()
