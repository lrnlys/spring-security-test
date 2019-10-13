package com.rinanz.configuration;

import lombok.extern.log4j.Log4j2;
import org.apache.sshd.client.SshClient;
import org.apache.sshd.client.channel.ChannelExec;
import org.apache.sshd.client.channel.ClientChannel;
import org.apache.sshd.client.channel.ClientChannelEvent;
import org.apache.sshd.client.future.ConnectFuture;
import org.apache.sshd.client.session.ClientSession;
import org.apache.sshd.server.SshServer;
import org.apache.sshd.server.auth.AsyncAuthException;
import org.apache.sshd.server.auth.password.PasswordAuthenticator;
import org.apache.sshd.server.auth.password.PasswordChangeRequiredException;
import org.apache.sshd.server.channel.ChannelSession;
import org.apache.sshd.server.command.Command;
import org.apache.sshd.server.command.CommandFactory;
import org.apache.sshd.server.keyprovider.SimpleGeneratorHostKeyProvider;
import org.apache.sshd.server.session.ServerSession;
import org.apache.sshd.server.shell.UnknownCommand;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.Future;

@Log4j2
@Configuration
public class SshConfiguration {

    private Integer port;

    public SshConfiguration(ApplicationProperties applicationProperties){
        this.port=3022;
    }

    @Bean
    public SshServer sshServer(){
        SshServer sshServer = SshServer.setUpDefaultServer();
        sshServer.setPort(this.port);
        sshServer.setKeyPairProvider(new SimpleGeneratorHostKeyProvider());
        sshServer.setPasswordAuthenticator(new PasswordAuthenticator(){

            @Override
            public boolean authenticate(String username, String password, ServerSession session) throws PasswordChangeRequiredException, AsyncAuthException {
                return true;
            }

        });
        sshServer.setCommandFactory(new CommandFactory() {
            @Override
            public Command createCommand(ChannelSession channel, String command) throws IOException {
                System.out.println(command);
                return new UnknownCommand(command);
            }
        });

        return sshServer;
    }

    public static void main(String[] args) throws IOException {
        String cmd="Test unkown command";
        SshClient client=SshClient.setUpDefaultClient();
        client.start();
        ConnectFuture connectFuture = client.connect("test", "localhost", 3022);
        connectFuture.await();
        ClientSession session=connectFuture.getSession();
        session.addPasswordIdentity("test");
        //session.addPublicKeyIdentity(SecurityUtils.loadKeyPairIdentity("keyname", new FileInputStream("priKey.pem"), null));
        if(!session.auth().await())
            System.out.println("auth failed");

        ChannelExec channelExec=session.createExecChannel(cmd);
        channelExec.setOut(System.out);
        channelExec.open();
        channelExec.waitFor(Arrays.asList(ClientChannelEvent.CLOSED), 0);
        channelExec.close();

        client.stop();
    }

}
