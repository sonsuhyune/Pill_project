<?php
//$con=mysqli_connect( "localhost", "pill", "pilldb", "pill_info");
//if(mysql_connect_errno($con)){
//    echo "Failed to connect to MySQL" . mysqli_connect_error();
//}

//mysqli_set_charset($con, "utf8");

include('dbcon.php');
$res = mysqli_query($con,"select * from TEST");

$result = array();

while($row = mysqli_fetch_array($res)){
        echo json_encode(array("result"=>$result));
        array_push($result,array('name'=>$row[0],'test1'=>$row[1],'test2'=>$row[2]));
}

echo json_encode(array("result"=>$result));

mysqli_close($con);
?>
