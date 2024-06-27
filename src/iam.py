import json

import boto3


from util import aws_account, check_response, on_laptop

SM_ROLE = "aws-llm-training-role"
SM_POLICY = "aws-llm-training-policy"

if on_laptop():
    SM_ROLE_ARN = "arn:aws:iam::" + aws_account() + ":role/" + SM_ROLE
    SM_POLICY_ARN = "arn:aws:iam::" + aws_account() + ":policy/" + SM_POLICY
else:
    SM_ROLE_ARN = "arn:aws:iam::" + "" + ":role/" + SM_ROLE
    SM_POLICY_ARN = "arn:aws:iam::" + "" + ":policy/" + SM_POLICY


def iam_setup(self):
    delete = True
    #
    client = boto3.client("iam")
    found = True
    policy_arn = ""
    try:
        response = client.get_role(RoleName=SM_ROLE)
    except Exception as e:  # noqa pylint:disable=broad-exception-caught
        assert e.__class__.__name__ == "NoSuchEntityException", "type: " + e.__class__.__name__
        found = False
        response = None
    if found:
        check_response(response)
        if delete:
            try:
                client.detach_role_policy(
                    RoleName=SM_ROLE,
                    PolicyArn=self.sm_policy_arn,
                )
            except Exception as e:  # noqa pylint:disable=broad-exception-caught
                assert e.__class__.__name__ == "NoSuchEntityException", (
                        "type: " + e.__class__.__name__)
            client.delete_role(RoleName=SM_ROLE)
            # print("delete_role():", to_json(response))
            check_response(response)
            found = False
    if not found:
        response = client.create_role(
            RoleName=SM_ROLE,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "sagemaker.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    },
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "AWS": "arn:aws:iam::" + aws_account() + ":user/didduran_laptop"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            })
        )
        # print("create_role():", to_json(response))
        check_response(response)
    #
    found = True
    try:
        response = client.get_policy(PolicyArn=self.sm_policy_arn)
    except Exception as e:  # noqa pylint:disable=broad-exception-caught
        assert e.__class__.__name__ == "NoSuchEntityException", "type: " + e.__class__.__name__
        found = False
    if found:
        # print("get_policy():", to_json(response))
        check_response(response)
        if delete:
            client.delete_policy(PolicyArn=self.sm_policy_arn)
            # print("delete_policy():", to_json(response))
            check_response(response)
            found = False
            check_response(response)
            policy_arn = ""
        else:
            policy_arn = response["Policy"]["Arn"]
    if not found:
        response = client.create_policy(
            PolicyName=SM_POLICY,
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "*",
                        "Resource": "*"
                    },
                ]
            })
        )
        # print("create_policy():", to_json(response))
        check_response(response)
        policy_arn = response["Policy"]["Arn"]
    self.assertNotEqual("", policy_arn)
    client.attach_role_policy(
        RoleName=SM_ROLE,
        PolicyArn=policy_arn
    )
