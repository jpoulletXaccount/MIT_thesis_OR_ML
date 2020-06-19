
import os,smtplib,ssl,sys
sys.path.append(os.getcwd().split("src")[0])

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.helpers import useful_paths, size_data,logging_files,local_path
from src.learn_create import coordinator


def send_email(msg):
        """
        Send an email when the program has either crashed or finished
        :param msg: the message content
        :return:
        """
        # Create a secure SSL context
        context = ssl.create_default_context()

        # set up the SMTP server
        s = smtplib.SMTP_SSL(host="smtp.gmail.com", port=465,context=context)
        s.login(local_path.EMAIL_SENDER_ADDRESS, local_path.EMAIL_PASSWORD)

        email = MIMEMultipart()
        email['From'] = local_path.EMAIL_SENDER_ADDRESS
        email['To'] = local_path.EMAIL_RECEIVER_ADDRESS
        email['Subject'] = msg
        email.attach(MIMEText(msg + local_path.NAME_MACHINE,'plain'))

        s.send_message(email)


if __name__ == '__main__':
    list_filename = []
    for r, d, f in os.walk(useful_paths.PATH_TO_BENCHMARK_CREATED_600):
        for file in f:
            if '.txt' in file:
                list_filename.append(os.path.join(r, file))
    print(len(list_filename), " benchmarks instances to be trained on")
    size_data.NUMBER_CUSTOMERS = 600

    # reset logs
    logging_files.reset_all()

    coor = coordinator.CoordinatorInterface(list_filename=list_filename,number_data_points=15000)
    coor.perform_creation_based_trees(from_points=False)

    send_email('Coordinator done ')


