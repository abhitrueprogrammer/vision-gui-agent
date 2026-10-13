import socket
import unittest
from unittest.mock import Mock, patch
from vision_gui_agent.gemini import GeminiClientPool


class GeminiTransportTests(unittest.TestCase):
    def pool(self):
        pool=object.__new__(GeminiClientPool)
        pool._keys=['fixture'];pool._index=0;pool.client=object()
        return pool

    def dns_error(self):
        error=RuntimeError('wrapped transport failure')
        error.__cause__=socket.gaierror(socket.EAI_AGAIN,'Temporary failure in name resolution')
        return error

    def test_temporary_dns_failure_retries_same_request_then_succeeds(self):
        request=Mock(side_effect=[self.dns_error(),'response'])
        pool=self.pool()
        with patch('time.sleep') as sleep:
            self.assertEqual(pool.generate(request),'response')
        sleep.assert_called_once_with(.5)
        self.assertEqual(request.call_count,2)
        self.assertIs(request.call_args.args[0],pool.client)

    def test_dns_retries_are_bounded_and_other_failures_are_not_retried(self):
        for failure,attempts in ((self.dns_error(),3),(TypeError('local serialization error'),1)):
            request=Mock(side_effect=failure)
            with self.subTest(attempts=attempts),patch('time.sleep'):
                with self.assertRaises(type(failure)):self.pool().generate(request)
                self.assertEqual(request.call_count,attempts)
