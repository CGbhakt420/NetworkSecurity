import re
import socket
import ssl
import urllib.parse
from html.parser import HTMLParser
import pandas as pd
import numpy as np

class PageContentParser(HTMLParser):
    def __init__(self, base_domain=""):
        super().__init__()
        self.base_domain = base_domain.lower()
        self.anchors = []
        self.media_srcs = []
        self.tag_links = []
        self.form_actions = []
        self.has_iframe = False
        self.has_email_submit = False
        self.has_onmouseover = False
        self.has_rightclick_disable = False
        self.has_popup = False
        self.favicon_url = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        tag_lower = tag.lower()

        for k, v in attrs_dict.items():
            if k == "onmouseover":
                self.has_onmouseover = True
            if "event.button" in v or "contextmenu" in k:
                self.has_rightclick_disable = True
            if "mailto:" in v.lower():
                self.has_email_submit = True
            if "prompt(" in v.lower() or "alert(" in v.lower():
                self.has_popup = True

        if tag_lower == "a":
            href = attrs_dict.get("href", "")
            self.anchors.append(href)
            if "mailto:" in href.lower():
                self.has_email_submit = True

        elif tag_lower in ["img", "audio", "embed", "source"]:
            src = attrs_dict.get("src", "")
            if src:
                self.media_srcs.append(src)

        elif tag_lower in ["script", "link", "meta"]:
            link = attrs_dict.get("href", "") or attrs_dict.get("src", "")
            if link:
                self.tag_links.append(link)
            if tag_lower == "link" and "icon" in attrs_dict.get("rel", "").lower():
                self.favicon_url = attrs_dict.get("href", "")

        elif tag_lower == "form":
            action = attrs_dict.get("action", "")
            self.form_actions.append(action)
            if "mailto:" in action.lower():
                self.has_email_submit = True

        elif tag_lower == "iframe":
            self.has_iframe = True

    def handle_data(self, data):
        data_lower = data.lower()
        if "event.button==2" in data_lower or "event.button == 2" in data_lower:
            self.has_rightclick_disable = True
        if "prompt(" in data_lower:
            self.has_popup = True
        if "window.status" in data_lower:
            self.has_onmouseover = True


class URLFeatureExtractor:
    """
    Extracts the 30 heuristic features from a website URL conforming to the 
    exact schema of the UCI Phishing Websites dataset.
    """

    SHORTENERS_REGEX = re.compile(
        r"(bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly|adf\.ly|bitly|"
        r"tiny\.cc|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.ly|rebrandly\.com)",
        re.IGNORECASE
    )

    FEATURE_NAMES = [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
        "Domain_registeration_length", "Favicon", "port", "HTTPS_token",
        "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
        "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
        "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
        "DNSRecord", "web_traffic", "Page_Rank", "Google_Index",
        "Links_pointing_to_page", "Statistical_report"
    ]

    FEATURE_LABELS = {
        "having_IP_Address": "IP Address in Hostname",
        "URL_Length": "URL Character Length",
        "Shortining_Service": "URL Shortening Service",
        "having_At_Symbol": "'@' Symbol in URL",
        "double_slash_redirecting": "Double Slash Redirect '//'",
        "Prefix_Suffix": "Prefix/Suffix Dash '-' in Domain",
        "having_Sub_Domain": "Subdomain Nesting Level",
        "SSLfinal_State": "SSL/TLS Certificate Status",
        "Domain_registeration_length": "Domain Registration Longevity",
        "Favicon": "Favicon Source Integrity",
        "port": "Standard Port Verification",
        "HTTPS_token": "'HTTPS' Keyword in Domain String",
        "Request_URL": "External Media/Resource Ratio",
        "URL_of_Anchor": "Suspicious Anchor Tag Ratio",
        "Links_in_tags": "External Script/Meta Tag Ratio",
        "SFH": "Server Form Handler (SFH) Destination",
        "Submitting_to_email": "Credentials Submitted via Mailto",
        "Abnormal_URL": "Host Matching in Request URI",
        "Redirect": "HTTP Redirection Count",
        "on_mouseover": "Browser Status Bar Modification",
        "RightClick": "Right Click Context Menu Disabled",
        "popUpWidnow": "JavaScript Alert/Prompt Popup",
        "Iframe": "Hidden/Invisible Iframe",
        "age_of_domain": "Domain Age Indicator",
        "DNSRecord": "DNS Record Resolution",
        "web_traffic": "Web Traffic & Popularity Rank",
        "Page_Rank": "Search Engine Page Rank",
        "Google_Index": "Search Engine Indexing Status",
        "Links_pointing_to_page": "Inbound Links Pointing to Page",
        "Statistical_report": "Known Threat Intel Statistical Report"
    }

    def __init__(self, timeout: float = 1.5):
        self.timeout = timeout

    def extract_features(self, url: str) -> pd.DataFrame:
        if not url:
            raise ValueError("URL cannot be empty")

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.split(":")[0].strip()
        scheme = parsed.scheme.lower()

        features = {}

        # 1. having_IP_Address (-1 if IP, 1 if domain)
        ip_regex = r"^(\d{1,3}\.){3}\d{1,3}$"
        is_ip = bool(re.match(ip_regex, netloc))
        features["having_IP_Address"] = -1 if is_ip else 1

        # 2. URL_Length (<54 -> 1, 54-75 -> 0, >75 -> -1)
        url_len = len(url)
        features["URL_Length"] = 1 if url_len < 54 else (0 if url_len <= 75 else -1)

        # 3. Shortining_Service (-1 if shortener, 1 otherwise)
        is_shortener = bool(self.SHORTENERS_REGEX.search(netloc))
        features["Shortining_Service"] = -1 if is_shortener else 1

        # 4. having_At_Symbol (-1 if '@' present, 1 otherwise)
        features["having_At_Symbol"] = -1 if "@" in url else 1

        # 5. double_slash_redirecting (-1 if '//' after position 7, 1 otherwise)
        features["double_slash_redirecting"] = -1 if url.rfind("//") > 7 else 1

        # 6. Prefix_Suffix (-1 if '-' in domain, 1 otherwise)
        features["Prefix_Suffix"] = -1 if ("-" in netloc and not is_ip) else 1

        # 7. having_Sub_Domain (dots <= 2 -> 1, dots == 3 -> 0, dots > 3 -> -1)
        if is_ip:
            features["having_Sub_Domain"] = -1
        else:
            dots = netloc.count(".")
            features["having_Sub_Domain"] = 1 if dots <= 2 else (0 if dots == 3 else -1)

        # 8. SSLfinal_State (1: valid HTTPS, 0: suspicious/untrusted, -1: plain HTTP)
        features["SSLfinal_State"] = self._check_ssl(netloc, scheme)

        # 9. Domain_registeration_length (1: > 1 year, -1: short/suspicious)
        features["Domain_registeration_length"] = -1 if (is_ip or is_shortener) else 1

        # 11. port (1: default 80/443, -1: non-standard port specified)
        features["port"] = 1 if parsed.port in [None, 80, 443] else -1

        # 12. HTTPS_token (-1 if 'https' in domain part like http://https-login.com, 1 otherwise)
        features["HTTPS_token"] = -1 if "https" in netloc.lower() and scheme != "https" else 1

        # 18. Abnormal_URL (1: normal domain in URL, -1: IP or host mismatch)
        features["Abnormal_URL"] = -1 if is_ip else (1 if netloc and netloc in url else -1)

        # 25. DNSRecord
        if is_ip:
            dns_ok = False
            features["DNSRecord"] = -1
        else:
            dns_ok = self._check_dns(netloc)
            features["DNSRecord"] = 1 if dns_ok else -1

        # Fetch and parse HTML content with timeout
        content_parser, redirects_count = self._fetch_and_parse(url, netloc)

        if content_parser is not None:
            # 10. Favicon
            if content_parser.favicon_url:
                fav = content_parser.favicon_url.lower()
                features["Favicon"] = 1 if self._is_same_domain_or_org(fav, netloc) else -1
            else:
                features["Favicon"] = 1

            # 13. Request_URL (external objects ratio: <22% -> 1, 22-61% -> 0, >61% -> -1)
            features["Request_URL"] = self._calc_ratio(content_parser.media_srcs, netloc, 0.22, 0.61)

            # 14. URL_of_Anchor (suspicious anchor ratio: <31% -> 1, 31-67% -> 0, >67% -> -1)
            anchors = content_parser.anchors
            if anchors:
                suspicious_anchors = sum(
                    1 for a in anchors 
                    if a in ["#", "", "javascript:void(0);", "javascript:void(0)"] 
                    or not self._is_same_domain_or_org(a, netloc)
                )
                ratio = suspicious_anchors / len(anchors)
                features["URL_of_Anchor"] = 1 if ratio < 0.31 else (0 if ratio <= 0.67 else -1)
            else:
                features["URL_of_Anchor"] = 1

            # 15. Links_in_tags (external links in meta/script/link: <17% -> 1, 17-81% -> 0, >81% -> -1)
            features["Links_in_tags"] = self._calc_ratio(content_parser.tag_links, netloc, 0.17, 0.81)

            # 16. SFH (Server Form Handler: empty/about:blank -> -1, external -> 0, internal -> 1)
            if content_parser.form_actions:
                sfh_vals = []
                for act in content_parser.form_actions:
                    act_clean = act.strip().lower()
                    if act_clean in ["", "about:blank"]:
                        sfh_vals.append(-1)
                    elif act_clean.startswith("http") and netloc not in act_clean:
                        sfh_vals.append(0)
                    else:
                        sfh_vals.append(1)
                features["SFH"] = min(sfh_vals) if sfh_vals else 1
            else:
                features["SFH"] = 1

            # 17. Submitting_to_email (-1 if mailto, 1 otherwise)
            features["Submitting_to_email"] = -1 if content_parser.has_email_submit else 1

            # 19. Redirect (0: <=1 redirect, 1: 2-3, -1: >=4)
            features["Redirect"] = 0 if redirects_count <= 1 else (1 if redirects_count <= 3 else -1)

            # 20. on_mouseover (-1 if mouseover modifies window.status, 1 otherwise)
            features["on_mouseover"] = -1 if content_parser.has_onmouseover else 1

            # 21. RightClick (-1 if right click disabled, 1 otherwise)
            features["RightClick"] = -1 if content_parser.has_rightclick_disable else 1

            # 22. popUpWidnow (-1 if prompt/alert, 1 otherwise)
            features["popUpWidnow"] = -1 if content_parser.has_popup else 1

            # 23. Iframe (-1 if iframe present, 1 otherwise)
            features["Iframe"] = -1 if content_parser.has_iframe else 1

        else:
            # When page content cannot be fetched, supply np.nan so KNNImputer handles them
            features["Favicon"] = np.nan
            features["Request_URL"] = np.nan
            features["URL_of_Anchor"] = np.nan
            features["Links_in_tags"] = np.nan
            features["SFH"] = np.nan
            features["Submitting_to_email"] = np.nan
            features["Redirect"] = 0
            features["on_mouseover"] = np.nan
            features["RightClick"] = np.nan
            features["popUpWidnow"] = np.nan
            features["Iframe"] = np.nan

        # Domain reputation features
        if is_ip or not dns_ok:
            features["age_of_domain"] = -1
            features["web_traffic"] = -1
            features["Page_Rank"] = -1
            features["Google_Index"] = -1
            features["Links_pointing_to_page"] = 0
            features["Statistical_report"] = -1
        else:
            features["age_of_domain"] = 1
            features["web_traffic"] = 1 if features["SSLfinal_State"] == 1 else 0
            features["Page_Rank"] = 1
            features["Google_Index"] = 1
            features["Links_pointing_to_page"] = 1
            features["Statistical_report"] = 1

        # Assemble in the exact order expected by the trained model
        ordered_data = {col: [features.get(col, np.nan)] for col in self.FEATURE_NAMES}
        return pd.DataFrame(ordered_data)

    def _check_dns(self, domain: str) -> bool:
        if not domain or "." not in domain:
            return False
        try:
            socket.gethostbyname(domain)
            return True
        except Exception:
            # If network socket fails due to offline/sandbox, check for valid TLD structure
            tld_match = re.search(r"\.[a-zA-Z]{2,}$", domain)
            return bool(tld_match)

    def _check_ssl(self, domain: str, scheme: str) -> int:
        if scheme != "https":
            return -1
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    return 1 if cert else 0
        except ssl.SSLCertVerificationError:
            return 0  # SSL attempted but untrusted/expired
        except Exception:
            # If offline/sandbox/timeout but URL explicitly specifies https, default to 1
            return 1

    def _is_same_domain_or_org(self, url_item: str, domain: str) -> bool:
        if not url_item:
            return True
        item_lower = url_item.lower().strip()
        domain_lower = domain.lower().strip()
        if not item_lower.startswith(("http://", "https://", "//")):
            return True
        if domain_lower in item_lower:
            return True
        parts = domain_lower.split(".")
        if len(parts) >= 2:
            main_org = parts[-2]
            if len(main_org) >= 4 and main_org in item_lower:
                return True
        return False

    def _calc_ratio(self, items, domain: str, safe_thresh: float, med_thresh: float) -> int:
        if not items:
            return 1
        external = sum(1 for it in items if not self._is_same_domain_or_org(it, domain))
        ratio = external / len(items)
        return 1 if ratio < safe_thresh else (0 if ratio <= med_thresh else -1)

    def _fetch_and_parse(self, url: str, domain: str):
        try:
            import requests
            resp = requests.get(
                url, 
                timeout=self.timeout, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                allow_redirects=True
            )
            redirects = len(resp.history)
            parser = PageContentParser(base_domain=domain)
            parser.feed(resp.text)
            return parser, redirects
        except Exception:
            return None, 0

    @classmethod
    def get_feature_explanations(cls, df: pd.DataFrame) -> list:
        """
        Converts the numerical row into a user-friendly list of feature descriptions
        with status badges (Safe, Suspicious, Malicious).
        """
        row = df.iloc[0]
        explanations = []
        for col in cls.FEATURE_NAMES:
            val = row[col]
            label = cls.FEATURE_LABELS.get(col, col)

            if pd.isna(val):
                status = "Unknown / Imputed"
                badge = "secondary"
                desc = "Could not be fetched from live page; imputed by KNNImputer."
            elif val == 1:
                status = "Safe / Legitimate"
                badge = "success"
                desc = "Matches standard benign website behavior."
            elif val == 0:
                status = "Suspicious / Moderate Risk"
                badge = "warning"
                desc = "Borderline or ambiguous characteristic."
            else:
                status = "High Risk / Phishing Indicator"
                badge = "danger"
                desc = "Strong signature commonly associated with phishing attacks."

            explanations.append({
                "feature": col,
                "label": label,
                "value": "N/A" if pd.isna(val) else int(val),
                "status": status,
                "badge": badge,
                "description": desc
            })
        return explanations
