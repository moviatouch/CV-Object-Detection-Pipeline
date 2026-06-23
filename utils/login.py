import os
import datetime
import requests

# Default token file path — override via TOKEN_PATH env var or by passing
# the path argument directly to the functions below.
_DEFAULT_TOKEN_PATH = os.getenv("ACCESS_TOKEN_PATH", "access_token.txt")

def get_custom_machine_setting(custom_machine_setting, vicki_app):
	while True:
		try:
			ret = requests.post(url=vicki_app, data="[\"FetchCustomMachineSetting\", \"{}\"]".format(custom_machine_setting)).json()["value"]
			return ret

		except Exception as e:
			print(e)
			continue

def get_custom_machine_settings(vicki_app, logger):
	if get_custom_machine_setting("environment", vicki_app) == "prod":
		base_url = get_custom_machine_setting("PROD_URL", vicki_app)
		logger.info('   MACHINE ENVIRONMENT: PROD')
	else:
		base_url = get_custom_machine_setting("TEST_URL", vicki_app)
		logger.info('   MACHINE ENVIRONMENT: DEV')
	# logger.info("   Fetching MACHINE ID...")
	machine_id = get_custom_machine_setting("machine_id", vicki_app)
	# logger.info("   Fetching MACHINE TOKEN...")
	machine_token = get_custom_machine_setting("machine_token", vicki_app)
	# logger.info("   Fetching MACHINE API Key...")
	machine_api_key = get_custom_machine_setting("machineAPIKey", vicki_app)
	# logger.info("   Logging into the MACHINE...")
	return base_url, machine_id, machine_token, machine_api_key

def login_as_machine(url, id, token, api_key, logger):
	try:
		headers = {"Content-Type": "application/json",
				   "grant_type": "client_credentials",
				   "apikey": api_key,
				   "machine_token": token}
		response = requests.post("{}/loyalty/machines/{}/login".format(url, id), headers=headers)
		if response.status_code == 200:
			# logger.info("   Login successuful")
			return response.json()['access_token']
		else:
			logger.info("   Login fail")
			logger.info(response)
			return -1
	except Exception as e:
		logger.info("   Error logging in as machine.")
		return -1

def write_access_token(access_token, path=None):
    path = path or _DEFAULT_TOKEN_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines([str(access_token) + '\n', datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")])

def read_access_token(path=None):
    path = path or _DEFAULT_TOKEN_PATH
    if not os.path.exists(path):
        return -1, -1
    with open(path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        return -1, -1
    access_token = lines[0].strip()
    if len(access_token) < 5:
        return -1, -1
    timestamp = datetime.datetime.strptime(lines[1].strip(), "%m/%d/%Y, %H:%M:%S")
    return access_token, timestamp

#regenerate new access token
def generate_access_token(base_url, machine_id, machine_token, machine_api_key, logger):
	# logger.info('   Generating new ACCESS TOKEN')
	access_token = login_as_machine(base_url, machine_id, machine_token, machine_api_key, logger)
	write_access_token(access_token)

#check if access token was generated within last 6 hours, otherwise regenerate
def get_current_access_token(base_url, machine_id, machine_token, machine_api_key, logger):
	# logger.info('   Looking for a stored ACCESS TOKEN...')   
	if not os.path.exists('access_token.txt'):
		# logger.info('   Failed: stored ACCESS TOKEN not found')
		generate_access_token(base_url, machine_id, machine_token, machine_api_key, logger)
	else:
		pass
		#logger.info('   Success: stored ACCESS TOKEN found')

	access_token, timestamp = read_access_token()

	if access_token == -1:
		generate_access_token(base_url, machine_id, machine_token, machine_api_key, logger)

	access_token, timestamp = read_access_token()

	# logger.info('   Checking ACCESS TOKEN timestamp...')
	if (datetime.datetime.now() - timestamp).total_seconds() / (60 * 60) > 6:
		# logger.info('   Failed: ACCESS TOKEN out of date')
		generate_access_token(base_url, machine_id, machine_token, machine_api_key, logger)
		access_token, timestamp = read_access_token()
	else:
		pass
		# logger.info('   Success: ACCESS TOKEN up to date')
	return access_token