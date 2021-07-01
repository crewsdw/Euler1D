import numpy as np
import scipy.special as sp
import tensorflow as tf
import cupy as cp

gl_nodes = {
    2: [-0.5773502691896257645092, 0.5773502691896257645092],
    3: [-0.7745966692414833770359, 0, 0.7745966692414833770359],
    4: [-0.861136311594052575224, -0.3399810435848562648027,
        0.3399810435848562648027, 0.861136311594052575224],
    5: [-0.9061798459386639927976, -0.5384693101056830910363, 0,
        0.5384693101056830910363, 0.9061798459386639927976],
    10: [-0.973906528517171720078, -0.8650633666889845107321,
         -0.6794095682990244062343, -0.4333953941292471907993,
         -0.1488743389816312108848, 0.1488743389816312108848,
         0.4333953941292471907993, 0.6794095682990244062343,
         0.8650633666889845107321, 0.973906528517171720078],
    32: [-0.997263861849481563545, -0.9856115115452683354002,
         -0.9647622555875064307738, -0.9349060759377396891709,
         -0.8963211557660521239653, -0.8493676137325699701337,
         -0.794483795967942406963, -0.7321821187402896803874,
         -0.6630442669302152009751, -0.5877157572407623290408,
         -0.5068999089322293900238, -0.421351276130635345364,
         -0.3318686022821276497799, -0.2392873622521370745446,
         -0.1444719615827964934852, -0.0483076656877383162348,
         0.048307665687738316235, 0.1444719615827964934852,
         0.2392873622521370745446, 0.33186860228212764978,
         0.4213512761306353453641, 0.5068999089322293900238,
         0.5877157572407623290408, 0.6630442669302152009751,
         0.7321821187402896803874, 0.7944837959679424069631,
         0.8493676137325699701337, 0.8963211557660521239653,
         0.9349060759377396891709, 0.9647622555875064307738,
         0.9856115115452683354002, 0.997263861849481563545],
    50: [-0.9988664044200710501855, -0.994031969432090712585,
         -0.985354084048005882309, -0.9728643851066920737133,
         -0.9566109552428079429978, -0.9366566189448779337809,
         -0.9130785566557918930897, -0.8859679795236130486375,
         -0.8554297694299460846114, -0.821582070859335948356,
         -0.784555832900399263905, -0.744494302226068538261,
         -0.70155246870682225109, -0.6558964656854393607816,
         -0.6077029271849502391804, -0.5571583045146500543155,
         -0.5044581449074642016515, -0.449806334974038789147,
         -0.3934143118975651273942, -0.335500245419437356837,
         -0.2762881937795319903276, -0.2160072368760417568473,
         -0.1548905899981459020716, -0.0931747015600861408545,
         -0.0310983383271888761123, 0.0310983383271888761123,
         0.09317470156008614085445, 0.154890589998145902072,
         0.2160072368760417568473, 0.2762881937795319903276,
         0.335500245419437356837, 0.3934143118975651273942,
         0.4498063349740387891471, 0.5044581449074642016515,
         0.5571583045146500543155, 0.60770292718495023918,
         0.6558964656854393607816, 0.7015524687068222510896,
         0.7444943022260685382605, 0.7845558329003992639053,
         0.8215820708593359483563, 0.8554297694299460846114,
         0.8859679795236130486375, 0.9130785566557918930897,
         0.9366566189448779337809, 0.9566109552428079429978,
         0.9728643851066920737133, 0.985354084048005882309,
         0.9940319694320907125851, 0.9988664044200710501855],
    100: [-0.9997137267734412336782, -0.9984919506395958184,
          -0.996295134733125149186, -0.993124937037443459652,
          -0.9889843952429917480044, -0.9838775407060570154961,
          -0.9778093584869182885538, -0.9707857757637063319309,
          -0.9628136542558155272937, -0.9539007829254917428493,
          -0.9440558701362559779628, -0.9332885350430795459243,
          -0.921609298145333952667, -0.9090295709825296904671,
          -0.8955616449707269866985, -0.8812186793850184155733,
          -0.8660146884971646234107, -0.8499645278795912842934,
          -0.8330838798884008235429, -0.815389238339176254394,
          -0.7968978923903144763896, -0.777627909649495475628,
          -0.757598118519707176036, -0.7368280898020207055124,
          -0.71533811757305644646, -0.6931491993558019659487,
          -0.670283015603141015803, -0.6467619085141292798326,
          -0.622608860203707771604, -0.5978474702471787212648,
          -0.5725019326213811913169, -0.546597012065094167468,
          -0.520158019881763056647, -0.493210789208190933569,
          -0.465781649773358042249, -0.437897402172031513109,
          -0.4095852916783015425289, -0.3808729816246299567634,
          -0.3517885263724217209723, -0.3223603439005291517225,
          -0.2926171880384719647376, -0.2625881203715034791689,
          -0.2323024818449739696495, -0.201789864095735997236,
          -0.1710800805386032748875, -0.1402031372361139732075,
          -0.1091892035800611150034, -0.0780685828134366366948,
          -0.046871682421591631615, -0.015628984421543082872,
          0.0156289844215430828722, 0.0468716824215916316149,
          0.0780685828134366366948, 0.1091892035800611150034,
          0.140203137236113973208, 0.1710800805386032748875,
          0.201789864095735997236, 0.23230248184497396965,
          0.262588120371503479169, 0.292617188038471964738,
          0.3223603439005291517225, 0.351788526372421720972,
          0.3808729816246299567634, 0.4095852916783015425289,
          0.437897402172031513109, 0.4657816497733580422492,
          0.4932107892081909335693, 0.5201580198817630566468,
          0.546597012065094167468, 0.5725019326213811913169,
          0.597847470247178721265, 0.6226088602037077716042,
          0.6467619085141292798326, 0.6702830156031410158026,
          0.6931491993558019659487, 0.71533811757305644646,
          0.7368280898020207055124, 0.7575981185197071760357,
          0.7776279096494954756276, 0.7968978923903144763896,
          0.815389238339176254394, 0.8330838798884008235429,
          0.8499645278795912842934, 0.8660146884971646234107,
          0.8812186793850184155733, 0.895561644970726986699,
          0.9090295709825296904671, 0.921609298145333952667,
          0.9332885350430795459243, 0.9440558701362559779628,
          0.953900782925491742849, 0.9628136542558155272937,
          0.9707857757637063319309, 0.9778093584869182885538,
          0.9838775407060570154961, 0.9889843952429917480044,
          0.993124937037443459652, 0.9962951347331251491861,
          0.9984919506395958184002, 0.9997137267734412336782]
}

gl_weights = {
    2: [1, 1],
    3: [0.5555555555555555555556, 0.8888888888888888888889,
        0.555555555555555555556],
    4: [0.3478548451374538573731, 0.6521451548625461426269,
        0.6521451548625461426269, 0.3478548451374538573731],
    5: [0.2369268850561890875143, 0.4786286704993664680413,
        0.5688888888888888888889, 0.4786286704993664680413,
        0.2369268850561890875143],
    10: [0.0666713443086881375936, 0.149451349150580593146,
         0.219086362515982043996, 0.2692667193099963550912,
         0.2955242247147528701739, 0.295524224714752870174,
         0.269266719309996355091, 0.2190863625159820439955,
         0.1494513491505805931458, 0.0666713443086881375936],
    32: [0.0070186100094700966004, 0.0162743947309056706052,
         0.0253920653092620594558, 0.0342738629130214331027,
         0.0428358980222266806569, 0.050998059262376176196,
         0.0586840934785355471453, 0.065822222776361846838,
         0.072345794108848506225, 0.0781938957870703064717,
         0.0833119242269467552222, 0.087652093004403811143,
         0.091173878695763884713, 0.09384439908080456563918,
         0.0956387200792748594191, 0.0965400885147278005668,
         0.0965400885147278005668, 0.0956387200792748594191,
         0.0938443990808045656392, 0.091173878695763884713,
         0.0876520930044038111428, 0.083311924226946755222,
         0.078193895787070306472, 0.072345794108848506225,
         0.065822222776361846838, 0.0586840934785355471453,
         0.0509980592623761761962, 0.0428358980222266806569,
         0.0342738629130214331027, 0.0253920653092620594558,
         0.0162743947309056706052, 0.0070186100094700966004],
    50: [0.002908622553155140958, 0.0067597991957454015028,
         0.0105905483836509692636, 0.0143808227614855744194,
         0.0181155607134893903513, 0.0217802431701247929816,
         0.02536067357001239044, 0.0288429935805351980299,
         0.0322137282235780166482, 0.0354598356151461541607,
         0.0385687566125876752448, 0.041528463090147697422,
         0.044327504338803275492, 0.0469550513039484329656,
         0.0494009384494663149212, 0.0516557030695811384899,
         0.0537106218889962465235, 0.05555774480621251762357,
         0.057189925647728383723, 0.058600849813222445835,
         0.05978505870426545751, 0.0607379708417702160318,
         0.06145589959031666375641, 0.0619360674206832433841,
         0.062176616655347262321, 0.062176616655347262321,
         0.0619360674206832433841, 0.0614558995903166637564,
         0.0607379708417702160318, 0.05978505870426545751,
         0.058600849813222445835, 0.057189925647728383723,
         0.055557744806212517624, 0.0537106218889962465235,
         0.05165570306958113849, 0.049400938449466314921,
         0.046955051303948432966, 0.044327504338803275492,
         0.0415284630901476974224, 0.0385687566125876752448,
         0.0354598356151461541607, 0.0322137282235780166482,
         0.02884299358053519803, 0.02536067357001239044,
         0.0217802431701247929816, 0.0181155607134893903513,
         0.0143808227614855744194, 0.010590548383650969264,
         0.0067597991957454015028, 0.0029086225531551409584],
    100: [7.346344905056717304e-4, 0.00170939265351810524,
          0.0026839253715534824194, 0.0036559612013263751823,
          0.0046244500634221193511, 0.005588428003865515157,
          0.006546948450845322764, 0.007499073255464711579,
          0.008443871469668971403, 0.009380419653694457951418,
          0.0103078025748689695858, 0.011225114023185977117,
          0.0121314576629794974077, 0.0130259478929715422856,
          0.013907710703718772688, 0.0147758845274413017689,
          0.0156296210775460027239, 0.0164680861761452126431,
          0.01729046056832358243934, 0.0180959407221281166644,
          0.0188837396133749045529, 0.0196530874944353058654,
          0.0204032326462094327668, 0.021133442112527641543,
          0.021843002416247386314, 0.0225312202563362727018,
          0.0231974231852541216225, 0.0238409602659682059626,
          0.02446120270795705272, 0.02505754448157958970376,
          0.025629402910208116076, 0.026176219239545676342,
          0.02669745918357096266, 0.0271926134465768801365,
          0.0276611982207923882942, 0.0281027556591011733176,
          0.0285168543223950979909, 0.0289030896011252031349,
          0.0292610841106382766201, 0.0295904880599126425118,
          0.0298909795933328309168, 0.03016226510516914491907,
          0.03040407952645482001651, 0.0306161865839804484965,
          0.0307983790311525904277, 0.030950478850490988234,
          0.031072337427566516588, 0.0311638356962099067838,
          0.0312248842548493577324, 0.0312554234538633569476,
          0.0312554234538633569476, 0.0312248842548493577324,
          0.031163835696209906784, 0.031072337427566516588,
          0.0309504788504909882341, 0.030798379031152590428,
          0.0306161865839804484965, 0.0304040795264548200165,
          0.030162265105169144919, 0.02989097959333283091684,
          0.029590488059912642512, 0.02926108411063827662,
          0.02890308960112520313488, 0.0285168543223950979909,
          0.0281027556591011733176, 0.027661198220792388294,
          0.0271926134465768801365, 0.02669745918357096266,
          0.026176219239545676342, 0.025629402910208116076,
          0.025057544481579589704, 0.02446120270795705272,
          0.0238409602659682059626, 0.023197423185254121622,
          0.0225312202563362727018, 0.021843002416247386314,
          0.02113344211252764154267, 0.020403232646209432767,
          0.0196530874944353058654, 0.0188837396133749045529,
          0.018095940722128116664, 0.0172904605683235824393,
          0.016468086176145212643, 0.0156296210775460027239,
          0.0147758845274413017689, 0.013907710703718772688,
          0.013025947892971542286, 0.0121314576629794974077,
          0.0112251140231859771172, 0.010307802574868969586,
          0.009380419653694457951, 0.008443871469668971402621,
          0.00749907325546471157883, 0.0065469484508453227642,
          0.0055884280038655151572, 0.0046244500634221193511,
          0.0036559612013263751823, 0.0026839253715534824194,
          0.00170939265351810524, 7.3463449050567173E-4]
}


class IRK:
    def __init__(self, order):
        self.order = int(order)
        self.nodes = np.array(self.get_nodes())
        self.weights = np.array(self.get_weights())
        self.weights_device = cp.asarray(self.weights)

        # build RK coefficient matrix
        self.rk_matrix = None
        self.rk_matrix_quads = None
        self.build_matrix()
        self.rk_matrix_tf32 = tf.convert_to_tensor(self.rk_matrix_quads.T, dtype=tf.float32)

    def get_nodes(self):
        nodes = gl_nodes.get(self.order, "nothing")
        return nodes

    def get_weights(self):
        weights = gl_weights.get(self.order, "nothing")
        return weights

    def build_matrix(self):
        self.rk_matrix_quads = np.zeros((self.order + 1, self.order + 1))
        self.rk_matrix_quads[:-1, :-1] = np.array([[0.5 * self.weights[i] * sum(self.series(s, i, j)
                                                                                for s in range(self.order))
                                                    + self.weights[i] / 2.0
                                                    for i in range(self.order)]
                                                   for j in range(self.order)])
        self.rk_matrix = cp.asarray(self.rk_matrix_quads[:-1, :-1])
        self.rk_matrix_quads[:-1, -1] = 0.5 * np.array(self.weights)

    def series(self, s, i, j):
        return 0.5 * sp.eval_legendre(s, self.nodes[i]) * (sp.eval_legendre(s + 1, self.nodes[j])
                                                           - sp.eval_legendre(s - 1, self.nodes[j]))
