import json
import requests


def send_alert(logger, vicki_app, message, warning = True):
    if warning:
        payload = json.dumps(["createMachineAlert", "CV:WARNING:" + message])
    else:
        payload = json.dumps(["createMachineAlert", "CV:INFO:" + message])
    headers = {'Content-Type': 'application/json'}
    logger.info(message)
    try:
        response = requests.post(vicki_app, headers=headers, data=payload)
        #self.logger.info(response)
        if response.status_code == 200:
            pass
            #self.logger.info('Sending alert - Success')
        else:
            logger.info('Sending alert - Failed')
    except Exception as e:
        logger.info(f'Sending alert - Failed: {e}')