import random
import csv
import os

# Configuration
NUM_SAMPLES_PER_CATEGORY = 200
OUTPUT_FILE = "/home/izumi/Documents/CODE/Chichi/data/alerts_dataset_v2.csv"

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
]

SUSPICIOUS_TEMPLATES = [
    "Notification: Login at {time} on {device} from {location}. Not you? Click {link}",
    "Account {account} was accessed from a new device. If this was not you, please contact support or click {link}",
    "Password change request for your {bank} mobile app received. Use code {otp} to continue.",
    "Your {bank} account profile was modified on {date}. If you did not make this change, visit {link}",
    "Security Alert: Your {bank} internet banking password was changed from {location}. Contact us if not you.",
    "Alert: A new device {device} has been linked to your {bank} profile. Not you? Secure account at {link}",
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
    }


def create_dataset():
    data = []

    # Generate Legitimate (0)
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        params = generate_random_data()
        template = random.choice(LEGITIMATE_TEMPLATES)
        data.append({"text": template.format(**params), "label": 0})

    # Generate Fake (1)
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        params = generate_random_data()
        template = random.choice(FAKE_TEMPLATES)
        data.append({"text": template.format(**params), "label": 1})

    # Generate Suspicious (2)
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        params = generate_random_data()
        template = random.choice(SUSPICIOUS_TEMPLATES)
        data.append({"text": template.format(**params), "label": 2})

    random.shuffle(data)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Generated {len(data)} samples in {OUTPUT_FILE}")


if __name__ == "__main__":
    create_dataset()
