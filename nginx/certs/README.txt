The certs in this directory (myCA.key/pem, rtstt.test.key/pem/crt) are
prebuilt test keys committed for convenience, so SSL works out of the box
for LAN/dev testing without generating anything yourself. Browser mic
access (getUserMedia) requires a secure context on any non-localhost
origin, which is why this exists at all.

You should probably replace these keys before relying on this long term —
reusing a publicly known private key is a security risk (see "Generating
your own keys" below).

## Trusting the cert in your browser

For a browser to accept `wss://` without a certificate error, it needs to
trust the cert one way or another:

1. No setup, single device: open `https://<host>:8002/` directly in a tab
   and click through the browser's untrusted-certificate warning
   ("Advanced" -> "Proceed"). That exception also covers the WebSocket
   connection to the same host, so nothing needs to be installed as a
   trusted CA.

2. No warnings, every site on a device: install `myCA.pem` as a trusted
   root (instructions below), or use mkcert
   (https://github.com/FiloSottile/mkcert) instead of the committed keys —
   it generates a local CA and installs it into your OS/browser trust
   stores automatically:
   ```
   mkcert -install
   mkcert -cert-file rtstt.test.pem -key-file rtstt.test.key rtstt.test <your-lan-ip>
   ```
   mkcert generates a fresh CA per machine instead of sharing one private
   key across everyone who clones this repo, which is the main downside of
   the committed `myCA.key`.

## Generating your own keys

# Linux instructions to generate your own key:
# Certificate Authority
openssl genrsa -des3 -out myCA.key 2048
openssl req -x509 -new -nodes -key myCA.key -sha256 -days 1825 -out myCA.pem
# HTTPS/SSL certificate
openssl genrsa -out rtstt.test.key 2048
openssl req -new -key rtstt.test.key -out rtstt.test.csr
nano rtstt.test.ext
# Sign and create the crt
openssl x509 -req -in rtstt.test.csr -CA myCA.pem -CAkey myCA.key -CAcreateserial -addtrust serverAuth -out rtstt.test.pem -days 825 -sha256 -extfile rtstt.test.ext


apt-get install ca-certificates
cp myCA.pem /usr/local/share/ca-certificates/
update-ca-certificates
