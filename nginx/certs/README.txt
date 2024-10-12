You should probably replace these keys. Trusting them long term is a significant security risk to you and your browser.

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
