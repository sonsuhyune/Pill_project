<?php
  
$connect=mysqli_connect( "localhost", "pill", "pilldb", "pill_info");

mysqli_query("SET NAMES UTF8");

session_start();

$id = $_POST[mark];

$sql = "SELECT name, img_path FROM pill_information WHERE name LIKE '%$id%'";

$result = mysqli_query($connect,$sql);

$data = array();

if($result){

    while($row=mysqli_fetch_array($result)){
        array_push($data,
            array('pill_name'=>$row['name'],
            'img'=>$row['img_path'] ));
    }

    header('Content-Type: application/json; charset=utf8');
    $json = json_encode(array("user_result"=>$data), JSON_PRETTY_PRINT+JSON_UNESCAPED_UNICODE);
    echo $json;

}
else{
    echo "SQL error : ";
    echo mysqli_error($connect);
}
?>