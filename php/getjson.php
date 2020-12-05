<?php
  
    error_reporting(E_ALL);
    ini_set('display_errors',1);

    include('dbcon.php');


    $stmt = $con->prepare('select * from pilltest');
    $stmt->execute();

    if ($stmt->rowCount() > 0)
    {
        $data = array();

        while($row=$stmt->fetch(PDO::FETCH_ASSOC))
        {
            extract($row);

            array_push($data,
                array('[34m~R~H목[34m~]m| [m~H[34m~Xm=>$4m~R~H목4m~]34m| [34m~H4m~X22;17H'[34m~R~H목m~E'=>$4m~R~H목34m~E,

                '[34m~W~E[34m~F~L[34m~]m| [m~H[34m~Xm=>$4m~W~E4m~F~L4m~]34m| [34m~H4m~X4;13H));

        }

        header('Content-Type: application/json; charset=utf8');
        $json = json_encode(array("webnautes"=>$data), JSON_PRETTY_PRINT+JSON_UNESCAPED_UNICODE);
        echo $json;
    }

?>