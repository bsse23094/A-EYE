"""Email tools — IMAP/SMTP via stdlib. Registered only when configured.

config.json:
    "email": {"imap_host": "imap.gmail.com", "smtp_host": "smtp.gmail.com",
              "user": "you@gmail.com", "password": "app-password"}
"""

from __future__ import annotations

import email
import email.header
import imaplib
import smtplib
from email.message import EmailMessage


def _decode(value: str) -> str:
    parts = email.header.decode_header(value or "")
    out = ""
    for text, enc in parts:
        out += text.decode(enc or "utf-8", errors="replace") if isinstance(text, bytes) else text
    return out


def _imap(cfg) -> imaplib.IMAP4_SSL:
    e = cfg.email
    conn = imaplib.IMAP4_SSL(e["imap_host"], int(e.get("imap_port", 993)))
    conn.login(e["user"], e["password"])
    return conn


def register(r) -> None:

    @r.register("email_unread", "List unread emails (sender, subject)",
                {"?limit": "integer: default 10"})
    def email_unread(ctx, limit: int = 10) -> str:
        n = max(1, min(25, int(limit or 10)))
        conn = _imap(ctx.cfg)
        try:
            conn.select("INBOX", readonly=True)
            _, data = conn.search(None, "UNSEEN")
            ids = data[0].split()
            if not ids:
                return "No unread emails."
            lines = [f"{len(ids)} unread email(s); latest {min(n, len(ids))}:"]
            for uid in reversed(ids[-n:]):
                _, msg_data = conn.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
                msg = email.message_from_bytes(msg_data[0][1])
                lines.append(f"  #{uid.decode()}: {_decode(msg['From'])} — "
                             f"{_decode(msg['Subject'])} ({msg['Date']})")
            return "\n".join(lines)
        finally:
            conn.logout()

    @r.register("email_read", "Read one email body by id (from email_unread)",
                {"id": "string: message id"})
    def email_read(ctx, id: str) -> str:
        conn = _imap(ctx.cfg)
        try:
            conn.select("INBOX", readonly=True)
            _, msg_data = conn.fetch(id.encode(), "(BODY.PEEK[])")
            if not msg_data or msg_data[0] is None:
                return f"No message with id {id}."
            msg = email.message_from_bytes(msg_data[0][1])
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(part.get_content_charset() or "utf-8",
                                                  errors="replace")
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(msg.get_content_charset() or "utf-8",
                                          errors="replace")
            if len(body) > 6000:
                body = body[:6000] + "\n...[truncated]"
            return (f"From: {_decode(msg['From'])}\nSubject: {_decode(msg['Subject'])}\n"
                    f"Date: {msg['Date']}\n\n{body or '(no plain-text body)'}")
        finally:
            conn.logout()

    @r.register("email_send", "Send an email",
                {"to": "string: recipient address",
                 "subject": "string: subject line",
                 "body": "string: message body"},
                needs_confirm=bool(r.ctx.cfg and r.ctx.cfg.confirm_email_send))
    def email_send(ctx, to: str, subject: str, body: str) -> str:
        e = ctx.cfg.email
        msg = EmailMessage()
        msg["From"] = e["user"]
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        port = int(e.get("smtp_port", 465))
        host = e.get("smtp_host", e["imap_host"].replace("imap", "smtp"))
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as s:
                s.login(e["user"], e["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls()
                s.login(e["user"], e["password"])
                s.send_message(msg)
        return f"Email sent to {to}."
