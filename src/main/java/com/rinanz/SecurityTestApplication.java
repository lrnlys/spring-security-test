package com.rinanz;

import lombok.extern.log4j.Log4j2;
import org.apache.sshd.server.SshServer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.ApplicationListener;
import org.springframework.context.ConfigurableApplicationContext;

import java.io.IOException;

@Log4j2
@SpringBootApplication
public class SecurityTestApplication implements ApplicationListener<ApplicationReadyEvent> {

	public static void main(String[] args) {
		SpringApplication.run(SecurityTestApplication.class, args);
	}

	@Override
	public void onApplicationEvent(ApplicationReadyEvent applicationReadyEvent) {
		ConfigurableApplicationContext configurableApplicationContext = applicationReadyEvent.getApplicationContext();
		SshServer sshServer = configurableApplicationContext.getBean(SshServer.class);
		try {
			sshServer.start();
		} catch (IOException e) {
			System.exit(1);
		}
	}
}
