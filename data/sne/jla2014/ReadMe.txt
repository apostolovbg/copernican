J/A+A/568/A22     Joint analysis of the SDSS-II and SNLS SNe Ia (Betoule+, 2014)
================================================================================
Improved cosmological constraints from a joint analysis of the SDSS-II
and SNLS supernova samples.
    Betoule M., Kessler R., Guy J., Mosher J., Hardin D., Biswas R., Astier P.,
    El-Hage P., Konig M., Kuhlmann S., Marriner J., Pain R., Regnault N.,
    Balland C., Bassett B.A., Brown P.J., Campbell H., Carlberg R.G.,
    Cellier-Holzem F., Cinabro D., Conley A., D'Andrea C.B., DePoy D.L., Doi M.,
    Ellis R.S., Fabbro S., Filippenko A.V., Foley R.J., Frieman J.A.,
    Fouchez D., Galbany L., Goobar A., Gupta R.R., Hill G.J., Hlozek R.,
    Hogan C.J., Hook I.M., Howell D.A., Jha S.W., Le Guillou L., Leloudas G.,
    Lidman C., Marshall J.L., Moller A., Mourao A.M., Neveu J., Nichol R.,
    Olmstead M.D., Palanque-Delabrouille N., Perlmutter S., Prieto J.L.,
    Pritchet C.J., Richmond M., Riess A.G., Ruhlmann-Kleider V., Sako M.,
    Schahmaneche K., Schneider D.P., Smith M., Sollerman J., Sullivan M.,
    Walton N.A., Wheeler C.J.
   <Astron. Astrophys. 568, A22 (2014)>
   =2014A&A...568A..22B
================================================================================
ADC_Keywords: Supernovae ; Redshifts ; Photometry
Keywords: cosmology: observations - distance scale - dark energy

Abstract:
    We deliver luminosity-distance measurements from a joint analysis of
    740 type-Ia supernovae from the SDSS and SNLS supernova surveys.

Description:
    The release consists in:
    1. The light-curve parameters for the 740 SNe Ia, obtained from a fit
       of the measured light-curve with the SALT2 v2.4 light-curve model.
       (tablef3.dat)
    2. The covariance matrix of the light-curve parameter uncertainties,
       including both the statistical and the systematic uncertainty. The
       ordering of parameters in the file is mb, x1 and color for the
       first SN, then mb, x1 and color for the second, etc.
       (tablef3.fit)
    3. Distance moduli obtained from a fit to the Light-curve parameters,
       as described in Appendix E of the paper (tablef1.dat)
    4. The covariance matrix of fitted distance moduli, including both
       statistical and systematic contribution (tablef2.fit)

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe          80        .   This file
tablef1.dat     19       31   Binned distance moduli
tablef2.fit   2880        4   Binned distance moduli (31x31) covariance matrix
tablef3.dat    193      740   Supernovae light-curve parameters
tablef4.fit   2880    13691   Light-curve parameter systematic (2220x2220) 
                              covariance matrix
--------------------------------------------------------------------------------

See also:
 J/A+A/552/A124 : SNLS & SDSS SN surveys photometric calibration (Betoule+ 2013)

Byte-by-byte Description of file: tablef1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label   Explanations
--------------------------------------------------------------------------------
   1-  5  F5.3  ---     z       [0.01/1.3] Redshift of the node
  10- 19  F10.7 mag     mu      [32.9/44.9] Interpolated distance (1)
--------------------------------------------------------------------------------
Note (1): The overall normalization of the distance moduli is arbitrary. See
  Appendix E for the exact meaning of these values and their usage in cosmology.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tablef3.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label      Explanations
--------------------------------------------------------------------------------
   1- 12  A12   ---      Name       Supernovae name
  13- 21  F9.7  ---      zcmb       [0,1.4] CMB frame redshift (1)
  22- 30  F9.7  ---      zhel       [0,1.4] Heliocentric redshift
      31  I1    ---    e_z          [0] Error in redshift (2)
  32- 41  F10.7 mag      mb         B band peak magnitude m_B_ (3)
  42- 50  F9.7  mag    e_mb         Error in mb (4)
  51- 60  F10.7 ---      x1         SALT2 shape (stretch) parameter x_1_
  61- 69  F9.7  ---    e_x1         Error in x1
  70- 79  F10.7 ---      c          SALT2 color parameter c
  80- 88  F9.7  ---    e_c          Error in color
  89- 98  F10.7 [Msun]   logMst     log_10_ host stellar mass
  99-108  F10.7 [Msun] e_logMst     Error in logMst
 109-121  F13.7 d        tmax       MJD corresponding to of peak brightness
 122-130  F9.7  d      e_tmax       Error in tmax
 131-140  F10.7 ---      cov(mb,s)  The covariance between mb and x1
 141-150  F10.7 ---      cov(mb,c)  The covariance between mb and colour
 151-160  F10.7 ---      cov(s,c)   The covariance between x1 and colour
     161  I1    ---      set        [1,4] Supernova sample identifier (5)
 162-172  F11.7 deg      RAdeg      Right Ascension in degree (J2000)
 173-183  F11.7 deg      DEdeg      Declination in degree (J2000)
 184-193  F10.7 mag      bias       The correction for analysis bias
--------------------------------------------------------------------------------
Note (1): The analysis includes a correction for the peculiar motions of nearby
  supernova based on the models of M.J. Hudson taken from the original low-z
  compilation (Conley et al. 2011ApJS..192....1C, Cat. J/ApJS/192/1).
Note (2): The analysis assumes a constant redshift uncertainty dominated by
  peculiar velocities of 150km/s rms, treated as an uncertainty on magnitudes.
Note (3): The distance modulus for a particular SN Ia in the SALT2 model is
  given by {mu}=m_B_-(M_B_-{alpha}x_1_+{beta}c), where x_1_ (time stretching
  parameter), c (color), and m_B_ (oberved peak magnitude in rest-frame B)
  are obtained by fitting its light curve. The reported mb values are corrected
  for selection bias given in column "bias".
Note (4): The reported error includes contributions from redshift 
  uncertainties, intrinsic SNe dispersion and lensing.
Note (5): Supernova sample identifiers as follows:
   1 = SNLS (Supernova Legacy Survey)
   2 = SDSS (Sloan Digital Sky Survey: SDSS-II SN Ia sample)
   3 = lowz (from CfA; Hicken et al. 2009, J/ApJ/700/331
   4 = Riess HST (2007ApJ...659...98R)
--------------------------------------------------------------------------------

Acknowledgements:
    Marc Betoule, marc.betoule(at)lpnhe.in2p3.fr

================================================================================
(End)          Marc Betoule [LPNHE, France], Patricia Vannier [CDS]  03-Jun-2014
