from sklearn.externals.joblib import Parallel, delayed
import time

class A( object ):

    def __init__( self, x ):
        self.x = x
        self.y = "Defined on .__init__()"

    def p(        self ):
        self.y = self.x**2

def aNormalFUN( aValueOfX ):
    time.sleep( float( aValueOfX ) / 10. )
    print ": aNormalFUN() has got aValueOfX == {0:} to process.".format( aValueOfX )
    return aValueOfX * aValueOfX

def aContainerFUN( aPayloadOBJECT ):
    time.sleep( float( aPayloadOBJECT.x ) / 10. )
    # try: except: finally:
    pass;  aPayloadOBJECT.p()
    print  "| aContainerFUN: has got aPayloadOBJECT.id({0:}) to process. [ Has made .y == {1:}, given .x == {2: } ]".format( id( aPayloadOBJECT ), aPayloadOBJECT.y, aPayloadOBJECT.x )
    time.sleep( 1 )

if __name__ == '__main__':
     # ------------------------------------------------------------------
     results = Parallel( n_jobs = 2
                         )(       delayed( aNormalFUN )( aParameterX )
                         for                             aParameterX in range( 11, 21 )
                         )
     print results
     print '.'
     # ------------------------------------------------------------------
     pass;       runs = [ A( x ) for x in range( 11, 21 ) ]
     # >>> type( runs )                        <type 'list'>
     # >>> type( runs[0] )                     <class '__main__.A'>
     # >>> type( run.p() for run in runs )     <type 'generator'>

     Parallel( verbose = 10,
               n_jobs  = 2
               )(        delayed( aContainerFUN )( run )
               for                                 run in runs
               )
